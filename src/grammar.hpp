//
// Copyright (c) 2020 Sergiu Deitsch <sergiu.deitsch@gmail.com>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//  * Neither the name of %ORGANIZATION% nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef PAV1IET_GRAMMAR_HPP
#define PAV1IET_GRAMMAR_HPP

#include <opencv2/core/core.hpp>

#include <string>
#include <tuple>
#include <vector>
#include <filesystem>

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/spirit/home/x3.hpp>

#include "adapted.hpp"

namespace pascal_v1 {

namespace ast {

struct Object
{
    unsigned id;
    std::string name;
    std::string label;
    cv::Point centerPoint;
    cv::Rect boundingBox;
};

struct Annotations
{
    std::filesystem::path imageFileName;
    cv::Size imageSize;
    int channels;
    std::string database;
    // Objects with ground truth
    std::vector<std::string> objectNames;
    cv::Point topLeft;
    std::vector<Object> objects;
};

} // namespace ast

using namespace boost::spirit::x3;

const rule<struct Comment> comment = "comment";
const rule<struct CommentStart> comment_start = "comment start";
const rule<struct Database, std::string> database = "database";
const rule<struct Header> header = "header";
const rule<struct ImageFileName, std::string> imageFileName = "image filename";
const rule<struct ImageSize, std::tuple<int, int, int> > imageSize = "image size";
const rule<struct Objects, std::vector<std::string> > objects = "objects";
const rule<struct Point, cv::Point> point = "point";
const rule<struct QuotedString, std::string> quoted_string = "quaoted string";
const rule<struct TopLeftCoordinate, cv::Point> top_left_coordinate = "top left coordinate";

const auto point_def = lit('(') >> int_ >> ',' >> int_ >> ')';
const auto comment_start_def = lit('#');
const auto quoted_string_def = '"' >> lexeme[+(char_ - '"')] >> '"';
const auto header_def = comment_start >> lit("PASCAL Annotation Version 1.00");
const auto imageFileName_def = lit("Image filename :") >> quoted_string;
const auto imageSize_def = lit("Image size (X x Y x C) :") >> int_ >> 'x' >> int_ >> 'x' >> int_;
const auto database_def = lit("Database :") >> quoted_string;
const auto objects_def = lit("Objects with ground truth :") >> omit[uint_] >> '{' >> +quoted_string >> '}';
const auto top_left_coordinate_def = comment_start >> "Top left pixel co-ordinates :" >> point;
const auto comment_def = comment_start >> *(char_ - comment_start);

BOOST_SPIRIT_DEFINE(comment);
BOOST_SPIRIT_DEFINE(comment_start);
BOOST_SPIRIT_DEFINE(database);
BOOST_SPIRIT_DEFINE(header);
BOOST_SPIRIT_DEFINE(imageFileName);
BOOST_SPIRIT_DEFINE(imageSize);
BOOST_SPIRIT_DEFINE(objects);
BOOST_SPIRIT_DEFINE(point);
BOOST_SPIRIT_DEFINE(quoted_string);
BOOST_SPIRIT_DEFINE(top_left_coordinate);

const rule<struct Rect, cv::Rect> rect = "rect";
const rule<struct OriginalLabel, std::tuple<unsigned, std::string, std::string> > original_label = "original label";
const rule<struct CenterPoint, std::tuple<unsigned, std::string, cv::Point> > center_point = "center point";
const rule<struct BoundingBox, std::tuple<unsigned, std::string, cv::Rect> > bounding_box = "bounding box";

const auto rect_def =
    (
           point
        >> '-'
        >> point
    )
    [
        (
            [] (auto& ctx)
            {
                cv::Point tl = boost::fusion::at_c<0>(_attr(ctx));
                cv::Point br = boost::fusion::at_c<1>(_attr(ctx));

                _val(ctx) = cv::Rect{tl, br};
            }
        )
    ]
    ;

const auto original_label_def =
    lit("Original label for object") >> uint_
    >> quoted_string >> ':' >> quoted_string;

const auto center_point_def =
    lit("Center point on object") >> uint_
    >> quoted_string >> "(X, Y)" >> ':' >> point;

const auto bounding_box_def =
    lit("Bounding box for object") >> uint_
    >> quoted_string >> "(Xmin, Ymin) - (Xmax, Ymax)" >> ':' >> rect;

BOOST_SPIRIT_DEFINE(rect);
BOOST_SPIRIT_DEFINE(original_label);
BOOST_SPIRIT_DEFINE(center_point);
BOOST_SPIRIT_DEFINE(bounding_box);

const auto object_def =
    (
           original_label
        >> center_point
        >> bounding_box
    )
    [
        (
            [] (auto& ctx)
            {
                const auto& label = boost::fusion::at_c<0>(_attr(ctx));
                const auto& center = boost::fusion::at_c<1>(_attr(ctx));
                const auto& bb = boost::fusion::at_c<2>(_attr(ctx));

                // TODO Ensure IDs an tags match

                _val(ctx).id = std::get<unsigned>(label);
                _val(ctx).name = std::get<1>(label);
                _val(ctx).label = std::get<2>(label);
                _val(ctx).centerPoint = std::get<cv::Point>(center);
                _val(ctx).boundingBox = std::get<cv::Rect>(bb);
            }
        )
    ]
    ;

const rule<struct Object, ast::Object> object = "object";

BOOST_SPIRIT_DEFINE(object);

const auto annotation_def =
       header
    >> imageFileName
    [
        (
            [] (auto& ctx)
            {
                _val(ctx).imageFileName = _attr(ctx);
            }
        )
    ]
    >> imageSize
    [
        (
            [] (auto& ctx)
            {
                cv::Size& imageSize = _val(ctx).imageSize;
                std::tie(imageSize.width, imageSize.height, _val(ctx).channels) = _attr(ctx);
            }
        )
    ]
    >> database
    [
        (
            [] (auto& ctx)
            {
                _val(ctx).database = _attr(ctx);
            }
        )
    ]
    >> objects
    [
        (
            [] (auto& ctx)
            {
                _val(ctx).objectNames = _attr(ctx);
            }
        )
    ]
    >> *(char_ - top_left_coordinate)
    >> top_left_coordinate
    [
        (
            [] (auto& ctx)
            {
                _val(ctx).topLeft = _attr(ctx);
            }
        )
    ]
    >>
    (
        +(
               omit[*(char_ - object)]
            >> object
        )
    )
    [
        (
            [] (auto& ctx)
            {
                _val(ctx).objects = _attr(ctx);
            }
        )
    ]
    ;

const rule<struct Annotation, ast::Annotations> annotation = "annotation";

BOOST_SPIRIT_DEFINE(annotation);

} // namespace pascal_v1

#endif // PAV1IET_GRAMMAR_HPP
