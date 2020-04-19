//
// pav1iet - PASCAL Annotation Version 1.00 Extractor Tool
// Copyright (C) 2020 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PAV1IET_ADAPTED_HPP
#define PAV1IET_ADAPTED_HPP

#include <opencv2/core/core.hpp>

#include <boost/fusion/adapted/struct/adapt_struct.hpp>

BOOST_FUSION_ADAPT_TPL_STRUCT((T), (cv::Point_) (T),
    (T, x)
    (T, y)
)

#endif // PAV1IET_ADAPTED_HPP
