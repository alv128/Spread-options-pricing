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

'''
vol1 = 0.10
vol2 = 0.15
corr = 0.3
strikes = K = 5.0
exp = 1.0
S1 = 109.998
S2 = 100
disc = 0.05
div1 = 0.03
div2 = 0.02


F1 = S1 * exp([disc-div1]*exp) = 109.998 * exp([0.05-0.03]*1.0) = 109.998 * exp(0.02) = 109.998 * 1.02020134003 = 112.220107
F2 = S2 * exp([disc-div2]*exp) = 100 * exp([0.05-0.02]*1.0) = 100 * exp(0.03) = 100 *. 1.03045453395 = 103.045453395

SQRT_VAR_EFF = vol2 * F2/(F2+K) = 0.15 * 103.04/(103.04 + 5.0) = 0.15 * 0.95372317998 = 0.14305812661977044

SQRT_VAR_ = sqrt(vol1**2 - 2 * corr * SQRT_VAR_EFF + SQRT_VAR_EFF**2) = sqrt(0.10**2 - 2*0.3*0.14305812661977044 + 0.14305812661977044**2) = sqrt(-0.05536924837990398) = sqrt(0.06722339079) = 0.25927473997

SQRT_VAR_ = sqrt(vol1**2 - 2 * vol1 * corr * SQRT_VAR_EFF + SQRT_VAR_EFF**2) = sqrt(0.1**2 - 2*0.1*0.3*0.14305812661977044 + 0.14305812661977044**2) = sqrt(0.021882139994772044) = 0.14792613019602738

SQRT_VAR = SQRT_VAR_ * sqrt(expiries) = 0.25927473997 * exp(1) = 0.14792613019602738 * 2.71828182846 = 0.40210491166612805

d1 = log(F1/(F2+K))/SQRT_VAR + SQRT_VAR/2 = log(112.220107/(103.045453395 + 5.0))/0.40210491166612805 + 0.40210491166612805/2
 = 0.03791018045259636/0.40210491166612805 + 0.40210491166612805/2 = 0.09427932699333347 +0.20105245583306403
= 0.2953317828263975

d2 = d1 - SQRT_VAR = 0.2953317828263975 - 0.40210491166612805 = -0.10677312883973056

discount_factors = tf.exp(-discount_rates * expiries)
D = exp(-0.05*1) = 0.951229424500714

N(d1) = 0.6161297802352117
N(d2) = 0.4574844828316779

D[F1 * N(d1) - (F2 + K) * N(d2)] = 0.951229424500714*(112.220107*0.6161297802352117 - (103.045453395+5.0)*0.4574844828316779) = 18.75161560430189


'''


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
    
    expected_price = np.array([18.7516])

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
    
    expected_price = np.array([18.7516])
    
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
      
      expected_price = np.array([21.6089])#TODO: calculate expected price for these params

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



