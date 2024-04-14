//
// pav1iet - PASCAL Annotation Version 1.00 Extractor Tool
// Copyright (C) 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
//
// This file is part of pav1iet.
//
// pav1iet is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// pav1iet is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with pav1iet.  If not, see <http://www.gnu.org/licenses/>.
//

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stop_token>
#include <thread>
#include <tuple>

// Enable for debugging purposes
// #define BOOST_SPIRIT_X3_DEBUG
#define BOOST_SPIRIT_X3_UNICODE

#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/program_options.hpp>
#include <boost/scope_exit.hpp>
#include <boost/spirit/home/support/iterators/istream_iterator.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/spirit/home/x3/support/utility/utf8.hpp>

#include <tbb/parallel_pipeline.h>

#include "grammar.hpp"

namespace {

constexpr const char* const banner =
    "PASCAL Annotation Version 1.00 Image Extraction Tool\n"
    "Copyright (C) 2024 Sergiu Deitsch\n"
    ;

void usage(const boost::program_options::options_description& desc)
{
    std::cout
        << banner <<
        "\n"
        "usage: pav1iet [options] [input]"
        "\n"
        << desc <<
        "\n"
        ;
}

void help(const boost::program_options::options_description& desc)
{
    usage(desc);

    std::cout <<
        "\n"
        "Report bugs to: sergiu.deitsch@gmail.com\n"
        ;
}

void version()
{
    std::cout << banner;
}

int processListing(std::istream& in, const std::filesystem::path& directory,
                   const std::filesystem::path& outBaseFileName)
{
    std::atomic_size_t emptyDescCount{0};
    std::atomic_size_t failCount{0};
    std::atomic_size_t numProcessedFiles{0};
    std::atomic_size_t numTotalFiles{0};
    std::atomic_size_t numObjects{0};
    std::atomic_size_t numWrittenImages{0};

    // Progress report thread
    std::jthread t
    (
        [&numProcessedFiles, &numTotalFiles, &numObjects] (std::stop_token token)
        {
            BOOST_SCOPE_EXIT(void)
            {
                std::clog << '\r' << '\n';
            }
            BOOST_SCOPE_EXIT_END

            using namespace std::chrono_literals;

            // Wait until the first line in the annotations listing has been
            // read.
            while (numTotalFiles.load(std::memory_order_relaxed) == 0) {
                std::this_thread::sleep_for(500ms);
            }

            while (!token.stop_requested()) {
                const std::size_t processed = numProcessedFiles.load(std::memory_order_relaxed);
                const std::size_t total = numTotalFiles.load(std::memory_order_relaxed);
                const std::size_t percent = processed * 100 / total;

                std::clog << '\r';
                std::clog << std::format(
                    "processed {0} out of {1} annotations ({3} objects) ({2}% "
                    "done)",
                    processed, total, percent,
                    numObjects.load(std::memory_order_relaxed));
                std::this_thread::sleep_for(500ms);

                if (percent >= 100) {
                    break;
                }
            }
        }
    );

    const auto readFileName = tbb::make_filter<void, std::filesystem::path>
    (
        tbb::filter_mode::serial_out_of_order,
        [&t, &in, &numTotalFiles, directory] (tbb::flow_control& fc)
        {
            std::string fileName;

            if (!std::getline(in, fileName)) {
                fc.stop();

                if (numTotalFiles.load(std::memory_order_relaxed) == 0) {
                    // In case no files could be read, directly notify the progress
                    // report thread to avoid a dead lock.
                    t.request_stop();
                }
            }
            else {
                numTotalFiles.fetch_add(1, std::memory_order_relaxed);
            }

            return directory / fileName;
        }
    );

    // Read in the annotations
    const auto loadAnnotations = tbb::make_filter<std::filesystem::path, std::tuple<std::filesystem::path, pascal_v1::ast::Annotations> >
    (
        tbb::filter_mode::serial_out_of_order,
        [] (const std::filesystem::path& fileName)
        {
            namespace x3 = boost::spirit::x3;

            std::ifstream in{fileName};
            in.unsetf(std::ios_base::skipws);

            pascal_v1::ast::Annotations annotations;

            // clang-format off
            bool parsed = x3::phrase_parse
            (
                  boost::spirit::istream_iterator{in}
                , boost::spirit::istream_iterator{}
                , pascal_v1::annotation >> x3::eoi
                , x3::unicode::space
                , annotations
            );
            // clang-format on

            if (!parsed) {
                throw std::runtime_error{"failed to parse annotations in " + fileName.string()};
            }

            return std::make_tuple(fileName, annotations);
        }
    );

    // Load images
    const auto loadImages = tbb::make_filter
    <
          std::tuple<std::filesystem::path, pascal_v1::ast::Annotations>
        , std::tuple<std::filesystem::path, pascal_v1::ast::Annotations, cv::Mat>
    >
    (
        tbb::filter_mode::serial_out_of_order,
        [&numObjects, directory] (const std::tuple<std::filesystem::path, pascal_v1::ast::Annotations>& t)
        {
            const auto& annotations = std::get<pascal_v1::ast::Annotations>(t);
            const std::filesystem::path imageFileName = directory / annotations.imageFileName;

            cv::Mat image = cv::imread(imageFileName.string());

            if (image.empty()) {
                throw std::invalid_argument{"failed to read image " + imageFileName.string()};
            }

            numObjects.fetch_add(annotations.objects.size(), std::memory_order_relaxed);

            return std::tuple_cat(t, std::make_tuple(image));
        }
    );

    const auto processObjects = tbb::make_filter
    <
          std::tuple<std::filesystem::path, pascal_v1::ast::Annotations, cv::Mat>
        , std::vector<cv::Mat>
    >
    (
        tbb::filter_mode::parallel,
        [&numProcessedFiles] (const std::tuple<std::filesystem::path, pascal_v1::ast::Annotations, cv::Mat>& t)
        {
            const auto& annotations = std::get<pascal_v1::ast::Annotations>(t);
            const cv::Mat& image = std::get<cv::Mat>(t);

            std::vector<cv::Mat> croppedImages;
            croppedImages.reserve(annotations.objects.size());

            const cv::Size windowSize{64, 128};
            const cv::Size padding{16, 16}; // one side
            const cv::Size padding2 = padding * 2; // all four sides

            for (const auto& object : annotations.objects) {
                cv::Mat patch;

                const cv::Rect rect = object.boundingBox;
                const cv::Point2f center = (rect.tl() + rect.br()) / 2.0f;

                cv::Size size = rect.size();

                cv::Size size1 = size;
                size1.height = size.width * windowSize.height / windowSize.width;

                cv::Size size2 = size;
                size2.width = size.height * windowSize.width / windowSize.height;

                assert(size1.height / size1.width == 2);
                assert(size2.height / size2.width == 2);

                // Compare ratios using integer arithmetic
                // i/j > k/l <=> il > kj
                const int ratio1 = size1.width * windowSize.height - size1.height * windowSize.width;

                // Use the ratio-corrected image (with the larger area)
                if (ratio1 < 0) {
                    assert(size1.area() >= size2.area());
                    size = size1;
                }
                else {
                    size = size2;
                }

                // Workout how much padding do we need to add to the original
                // bounding box such that we obtain the desired padding in the
                // resized image.
                cv::Size extraPadding2;
                extraPadding2.width = size.width * padding2.width / windowSize.width;
                extraPadding2.height = extraPadding2.width * windowSize.height / windowSize.width;

                cv::Size newSize = size + extraPadding2;

                int y = static_cast<int>(center.y);

                int topOverflow = y - newSize.height / 2;
                int bottomOverflow = image.rows - (y + newSize.height / 2);

                if (topOverflow < 0 || bottomOverflow < 0) {
                    // Cannot add sufficient vertical padding at the top/bottom
                    int paddingV = topOverflow < 0
                                       ? rect.y
                                       : image.rows - (rect.y + rect.height);

                    newSize.height = size.height + paddingV * 2;
                    // Account for added vertical padding
                    newSize.width =
                        newSize.height * windowSize.width / windowSize.height;
                }

                cv::Matx33f scale = cv::Matx33f::eye();
                scale(0, 0) = static_cast<float>(windowSize.width) /
                              static_cast<float>(newSize.width);
                scale(1, 1) = static_cast<float>(windowSize.height) /
                              static_cast<float>(newSize.height);

                cv::Matx33f translate = cv::Matx33f::eye();
                translate(0, 2) = -(center.x - static_cast<float>(newSize.width) / 2.0f);
                translate(1, 2) = -(center.y - static_cast<float>(newSize.height) / 2.0f);

                cv::Matx33f tmp = scale * translate;
                // Take the two top rows.
                cv::Mat1f M(2, 3, tmp.val);

                // In case we are downsampling, avoid antialiasing.
                cv::InterpolationFlags flags =
                    newSize.area() > windowSize.area() ? cv::INTER_AREA
                                                       : cv::INTER_CUBIC;
                cv::warpAffine(image, patch, M, windowSize, flags,
                               cv::BORDER_REFLECT);

                croppedImages.push_back(std::move(patch));
            }

            numProcessedFiles.fetch_add(1, std::memory_order_relaxed);

            return croppedImages;
        }
    );

    boost::format outFileNameFmt;

    if (boost::format tmp{outBaseFileName.string()}; tmp.expected_args() == 0) {
        outFileNameFmt = boost::format{outBaseFileName.string() + "%1%.png"};
    }
    else if (tmp.expected_args() > 1) {
        std::cerr << "error: output file name format must contain exactly one placeholder" << std::endl;
        return EXIT_FAILURE;
    }
    else {
        outFileNameFmt = std::move(tmp);
    }

    const auto writePatches = tbb::make_filter
    <
          std::vector<cv::Mat>
        , void
    >
    (
        tbb::filter_mode::serial_in_order,
        [&numWrittenImages, outFileNameFmt] (const std::vector<cv::Mat>& croppedImages)
        {
            boost::format fmt = outFileNameFmt;

            for (const cv::Mat& image : croppedImages) {
                cv::imwrite(str(fmt % numWrittenImages.fetch_add(1, std::memory_order_relaxed)), image);
            }
        }
    );

    try {
        tbb::parallel_pipeline
        (
              std::thread::hardware_concurrency()
            , readFileName
            & loadAnnotations
            & loadImages
            & processObjects
            & writePatches
        );

        // Wait until the progress report thread exists
        t.join();
    }
    catch (const std::exception& e) {
        t.request_stop();
        t.join();

        std::cerr << "error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (numWrittenImages > 0) {
        const std::filesystem::path tmp = outBaseFileName.parent_path();

        std::clog << std::format(
                         "wrote {} images to {}",
                         numWrittenImages.load(std::memory_order_relaxed),
                         (tmp.empty() ? std::filesystem::current_path() : tmp)
                             .string())
                  << std::endl;
    }

    if (in.bad()) {
        std::cerr << "error: an error occured while reading from input" << std::endl;
        return EXIT_FAILURE;
    }

    //if (emptyDescCount > 0) {
    //    std::clog << boost::format("warning: omitted %1% empty descriptor files") % emptyDescCount << std::endl;
    //}

    return failCount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char** argv)
{
    namespace po = boost::program_options;

    po::options_description opts{"available options"};

    std::filesystem::path fileName;
    std::filesystem::path outBaseFileName;

    opts.add_options()
        ("input,i", (po::value(&fileName))->value_name("<file>"), "annotations list file name")
        ("output,o", (po::value(&outBaseFileName))->value_name("<file>"), "output base file name")
        ("version,v", "show version information")
        ("help,h", "show this help message")
        ;

    po::positional_options_description pdesc;
    pdesc.add("input", 1);

    boost::program_options::variables_map vars;

    po::store(po::command_line_parser(argc, argv)
        .options(opts)
        .positional(pdesc)
        .run(), vars);

    if (vars.empty()) {
        usage(opts);
        return EXIT_FAILURE;
    }

    if (vars.count("help") != 0u) {
        help(opts);
        return EXIT_SUCCESS;
    }

    if (vars.count("version") != 0u) {
        version();
        return EXIT_SUCCESS;
    }

    po::notify(vars);

    if (fileName.empty()) {
        if (outBaseFileName.empty()) {
            std::cerr << "error: you must provide the output base file name" << std::endl;
            return EXIT_FAILURE;
        }

        // Read from stdin
        return processListing(std::cin, std::filesystem::current_path(), outBaseFileName);
    }

    if (outBaseFileName.empty()) {
        outBaseFileName = fileName.filename().replace_extension();
    }

    std::ifstream in{fileName};

    if (!in) {
        std::cerr << "error: failed to open " << fileName << std::endl;
        return EXIT_FAILURE;
    }

    return processListing(in, fileName.parent_path(), outBaseFileName);
}
