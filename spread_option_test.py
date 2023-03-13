# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for spread_option module."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from spread_option import spread_option_price
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


#Formula for call options:
#S1*N(d1) - (S2 + K*e^(-rt))*N(d2)
#D[F1 * N(d1) - (F2 + K) * N(d2)]
#d1 = log(F1/(F2+K))/SQRT_VAR + SQRT_VAR/2
#d2 = d1 - SQRT_VAR
#SQRT_VAR = SQRT_VAR_ * sqrt(expiries)
#SQRT_VAR_ = sqrt(volatilities1**2 - 2 * correlations * SQRT_VAR_EFF + SQRT_VAR_EFF**2)
#SQRT_VAR_EFF = volatilities2 * F2/(F2+K)
#F1 = S1 * exp((discount-dividends1)*expiry)
#F2 = S2 * exp((discount-dividends1)*expiry)


@test_util.run_all_in_graph_and_eager_modes
class SpreadOptionTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the spread option module."""

  def test_option_prices(self):
    volatilities1 = np.array([0.10])
    volatilities2 = np.array([0.15])
    correlations = np.array([0.3])
    strikes = np.array([5.0])
    expiries = 1.0
    spots1 = np.array([109.998])
    spots2 = np.array([100])
    discount_rates = np.array([0.05])
    dividend_rates1 = np.array([0.03])
    dividend_rates2 = np.array([0.02])
    
    expected_price = np.array([8.3636])

    computed_price = spread_option_price(
        volatilities1=volatilities1,
        volatilities2=volatilities2,
        correlations=correlations,
        strikes=strikes,
        expiries=expiries,
        spots1=spots1,
        spots2=spots2,
        discount_rates=discount_rates,
        dividend_rates1=dividend_rates1,
        divident_rates2=dividend_rates2,
    )
    
    self.assertAllClose(expected_price, computed_price, 1e-10)

  #implement test for scalar input as in asian options test
  def test_option_prices_scalar_input(self):
    volatilities1 = 0.10
    volatilities2 = 0.15
    correlations = 0.3
    strikes = 5.0
    expiries = 1.0
    spots1 = 109.998
    spots2 = 100
    discount_rates = 0.05
    dividend_rates1 = 0.03
    dividend_rates2 = 0.02
    
    expected_price = np.array([8.3636])
    
    computed_price = spread_option_price(
        volatilities1=volatilities1,
        volatilities2=volatilities2,
        correlations=correlations,
        strikes=strikes,
        expiries=expiries,
        spots1=spots1,
        spots2=spots2,
        discount_rates=discount_rates,
        dividend_rates1=dividend_rates1,
        divident_rates2=dividend_rates2,
    )

    self.assertAllClose(expected_price, computed_price, 1e-10)
  
  #implement test for margrabes formula when strike is 0 (K=0)
  #https://en.wikipedia.org/wiki/Margrabe%27s_formula
  def test_margrages_formula(self):
      volatilities1 = np.array([0.10])
      volatilities2 = np.array([0.15])
      correlations = np.array([0.3])
      strikes = np.array([0])
      expiries = 1.0
      spots1 = np.array([109.998])
      spots2 = np.array([100])
      discount_rates = np.array([0.05])
      dividend_rates1 = np.array([0.03])
      dividend_rates2 = np.array([0.02])
      
      expected_price = np.array([8.3636])#TODO: calculate expected price for these params

      computed_price = spread_option_price(
          volatilities1=volatilities1,
          volatilities2=volatilities2,
          correlations=correlations,
          strikes=strikes,
          expiries=expiries,
          spots1=spots1,
          spots2=spots2,
          discount_rates=discount_rates,
          dividend_rates1=dividend_rates1,
          divident_rates2=dividend_rates2,
      )

      self.assertAllClose(expected_price, computed_price, 1e-10)
    
if __name__ == '__main__':
    tf.test.main()



