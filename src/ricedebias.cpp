//
// Copyright (c) 2016-2017 Enrico Kaden & University College London
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>

#include "cartesianrange.h"
#include "darray.h"
#include "debug.h"
#include "diffenc.h"
#include "fitmcmicro.h"
#include "fmt.h"
#include "nifti.h"
#include "opts.h"
#include "parfor.h"
#include "progress.h"
#include "ricedebias.h"
#include "sarray.h"
#include "version.h"

static const char VERSION[] = R"(fitmcmicro)"
                              " " STR(SMT_VERSION_STRING);

static const char LICENSE[] = R"(
Copyright (c) 2016-2017 Enrico Kaden & University College London
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
)";

static const char USAGE[] = R"(
MULTI-COMPARTMENT MICROSCOPIC DIFFUSION IMAGING (SPHERICAL MEAN TECHNIQUE)

Copyright (c) 2016-2017 Enrico Kaden & University College London

If you use this software, please cite:
  Kaden E, Kelm ND, Carson RP, Does MD, and Alexander DC: Multi-
  compartment microscopic diffusion imaging. NeuroImage, 139:346–359,
  2016.  http://dx.doi.org/10.1016/j.neuroimage.2016.06.002

Usage:
  ricedebias [options] <input> <output>
  ricedebias (-h | --help)
  ricedebias --license
  ricedebias --version

Options:
  --mask <mask>        Foreground mask [default: none]
  --rician <rician>    Rician noise [default: none]
  --maxdiff <maxdiff>  Maximum diffusivity (mm²/s) [default: 3.05e-3]
  -h, --help           Help screen
  --license            License information
  --version            Software version
)";



template <typename float_t>
smt::inifti<float_t, 3> read_mask(std::map<std::string, docopt::value> &args)
{
  if (args["--mask"] && args["--mask"].asString() != "none")
  {
    return smt::inifti<float_t, 3>(args["--mask"].asString());
  }
  else
  {
    return smt::inifti<float_t, 3>();
  }
}

template <typename float_t>
std::tuple<float_t, smt::inifti<float_t, 3>> read_rician(std::map<std::string, docopt::value> &args)
{
  if (args["--rician"] && args["--rician"].asString() != "none")
  {
    std::istringstream sin(args["--rician"].asString());
    float_t scalar;
    if (!(sin >> scalar))
    {
      return std::make_tuple(float_t(0), smt::inifti<float_t, 3>(args["--rician"].asString()));
    }
    else
    {
      return std::make_tuple(scalar, smt::inifti<float_t, 3>());
    }
  }
  else
  {
    return std::make_tuple(float_t(0), smt::inifti<float_t, 3>());
  }
}

template <typename float_t>
float_t read_maxdiff(std::map<std::string, docopt::value> &args)
{
  if (args["--maxdiff"])
  {
    std::istringstream sin(args["--maxdiff"].asString());
    float_t maxdiff;
    if (!(sin >> maxdiff))
    {
      smt::error("Unable to parse ‘" + args["--maxdiff"].asString() + "’.");
      std::exit(EXIT_FAILURE);
    }
    else
    {
      return maxdiff;
    }
  }
  else
  {
    return float_t(3.05e-3);
  }
}


int main(int argc, const char **argv)
{

  typedef double float_t;

  // Input

  std::map<std::string, docopt::value> args = smt::docopt(USAGE, {argv + 1, argv + argc}, true, VERSION);
  if (args["--license"].asBool())
  {
    std::cout << LICENSE << std::endl;
    return EXIT_SUCCESS;
  }

  const smt::inifti<float_t, 4> input(args["<input>"].asString());


  const smt::inifti<float_t, 3> mask = read_mask<float_t>(args);
  if (mask)
  {
    if (input.size(0) != mask.size(0) || input.size(1) != mask.size(1) || input.size(2) != mask.size(2))
    {
      smt::error("‘" + args["<input>"].asString() + "’ and ‘" + args["--mask"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
    if (input.pixsize(0) != mask.pixsize(0) || input.pixsize(1) != mask.pixsize(1) || input.pixsize(2) != mask.pixsize(2))
    {
      smt::error("The pixel sizes of ‘" + args["<input>"].asString() + "’ and ‘" + args["--mask"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
    if (!input.has_equal_spatial_coords(mask))
    {
      smt::error("The coordinate systems of ‘" + args["<input>"].asString() + "’ and ‘" + args["--mask"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
  }

  const std::tuple<float_t, smt::inifti<float_t, 3>> rician = read_rician<float_t>(args);
  if (std::get<1>(rician))
  {
    if (input.size(0) != std::get<1>(rician).size(0) || input.size(1) != std::get<1>(rician).size(1) || input.size(2) != std::get<1>(rician).size(2))
    {
      smt::error("‘" + args["<input>"].asString() + "’ and ‘" + args["--rician"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
    if (input.pixsize(0) != std::get<1>(rician).pixsize(0) || input.pixsize(1) != std::get<1>(rician).pixsize(1) || input.pixsize(2) != std::get<1>(rician).pixsize(2))
    {
      smt::error("The pixel sizes of ‘" + args["<input>"].asString() + "’ and ‘" + args["--rician"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
    if (!input.has_equal_spatial_coords(std::get<1>(rician)))
    {
      smt::error("The coordinate systems of ‘" + args["<input>"].asString() + "’ and ‘" + args["--rician"].asString() + "’ do not match.");
      return EXIT_FAILURE;
    }
  }

  const float_t maxdiff = read_maxdiff<float_t>(args);

  // Processing

  smt::onifti<float, 4> output = smt::onifti<float, 4>(smt::format_string(args["<output>"].asString()), input, input.size(0), input.size(1), input.size(2), input.size(3));

  const unsigned int nthreads = smt::threads();
  const std::size_t chunk = 10;

  for (int zz = 0; zz < input.size(3); zz++)
  {
    for (int kk = 0; kk < input.size(2); kk++)
    {
      for (int jj = 0; jj < input.size(1); jj++)
      {
        for (int ii = 0; ii < input.size(0); ii++)
        {
          if ((!mask) || mask(ii, jj, kk) > 0)
          {
            output(ii, jj, kk, zz) = input(ii, jj, kk, zz);
            if (std::get<1>(rician))
            {
              output(ii, jj, kk, zz) = smt::ricedebias(input(ii, jj, kk, zz), std::get<1>(rician)(ii, jj, kk));
            }
            else
            {
              if (std::get<0>(rician) > float_t(0))
              {
                output(ii, jj, kk, zz) = smt::ricedebias(input(ii, jj, kk, zz), std::get<0>(rician));
              }
            }
            //output(ii, jj, kk, zz) = input_tmp(ll);
          }
          else
          {
            //output(ii, jj, kk, zz) = input_tmp(ll);
          }
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
