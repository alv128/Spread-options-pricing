# Copyright 2023 Google LLC
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

"""Analytical approximation for the spread-option price under Black-Scholes using Krik's approximation and WKB method.

## References

[1] C. F. Lo, 2013. A simple derivation of Kirk's approximation for spread options. Applied Mathematical Letters.
[2] D. Prathumwan & K. Trachoo, 2020. On the solution of two-dimensional fractional Black-Scholes Equation for 
    European put option. Advances in Difference Equations.

"""

import numpy as np
import tensorflow.compat.v2 as tf
#from tf_quant_finance.black_scholes import vanilla_prices

#spread option price: P_0(S_1, S_2, t) = S_1*N(d_1) - (S_2 + K*e^(-rt))*N(d_2)
#regular option price:       P_0(S, t) = S*N(d_1) - K*e^(-rt)*N(d_2)
#S = D*F, F = S/D forward price, D = e^(-rt)

#Kirk's approximation using WKB method for the two-dimensional Black-Scholes models for spread-options
def spread_option_price(volatilities1,
                        volatilities2,
                        correlations,
                        strikes,#K
                        expiries,#t
                        spots1,#S_1
                        spots2,#S_2
                        forwards1,
                        forwards2,
                        discount_rates,#r
                        dividend_rates1,#if provided, substract from discount_rates, discount_rates - dividend_rates
                        dividend_rates2,
                        discount_factors,#e^(-rt), mutually exclusive to discount_rates
                        is_call_options,#if not provided, assume call options
                        dtype=None,
                        name=None):

    if (spots1 is None) == (forwards1 is None):
        if (spots2 is None) == (forwards2 is None):
            raise ValueError('Either spots or forwards must be supplied but not both.')
        elif (spots2 is not None) or (forwards2 is not None):
            raise ValueError('Either spots or forwards for both assets must be supplied.')

    #if (spots is None) == (forwards is None):
    #    raise ValueError('Either spots or forwards must be supplied but not both.')

    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')
    
    with tf.name_scope(name or 'spread_option_price'):

        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = strikes.dtype
        volatilities1 = tf.convert_to_tensor(
            volatilities1, dtype=dtype, name='volatilities1')
        volatilities2 = tf.convert_to_tensor(
            volatilities2, dtype=dtype, name='volatilities2')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        correlations = tf.convert_to_tensor(correlations, dtype=dtype, name='correlations')

        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(
                discount_rates, dtype=dtype, name='discount_rates')
            discount_factors = tf.exp(-discount_rates * expiries)
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(
                discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = -tf.math.log(discount_factors) / expiries
        else:
            discount_rates = tf.convert_to_tensor(
                0.0, dtype=dtype, name='discount_rates')
            discount_factors = tf.convert_to_tensor(
                1.0, dtype=dtype, name='discount_factors')
        #if dividend_rates1 is None:
        if dividend_rates1 is not None:
            dividend_rates1 = tf.convert_to_tensor(
                dividend_rates1, dtype=dtype, name='dividend_rates1')
        else:
            dividend_rates1 = tf.convert_to_tensor(
                0.0, dtype=dtype, name='dividend_rates1')
        if dividend_rates2 is not None:
            dividend_rates2 = tf.convert_to_tensor(
                dividend_rates2, dtype=dtype, name='dividend_rates2')
        else:
            dividend_rates2 = tf.convert_to_tensor(
                0.0, dtype=dtype, name='dividend_rates2')

        #if forwards is not None:
        #    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        #else:
        #    spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
        #    forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)

        if forwards1 is not None and forwards2 is not None:
            forwards1 = tf.convert_to_tensor(forwards1, dtype=dtype, name='forwards1')
            forwards2 = tf.convert_to_tensor(forwards2, dtype=dtype, name='forwards2')
        else:
            spots1 = tf.convert_to_tensor(spots1, dtype=dtype, name='spots1')
            spots2 = tf.convert_to_tensor(spots2, dtype=dtype, name='spots2')
            forwards1 = spots1 * tf.exp((discount_rates - dividend_rates1) * expiries)
            forwards2 = spots2 * tf.exp((discount_rates - dividend_rates2) * expiries)
            
        #sqrt_var1 = volatilities1 * tf.math.sqrt(expiries)#sigma_s * sqrt(t)
        #sqrt_var2 = volatilities2 * tf.math.sqrt(expiries)

        #sqrt_var_eff = sqrt_var2 * (forwards2 / (forwards2 + strikes)) #This is the formula
        sqrt_var_eff = volatilities2 * tf.math.divide_no_nan(forwards2, (forwards2 + strikes))#is no nan needed?
        #volalities are not squared
        sqrt_var_ = tf.math.sqrt(tf.math.square(volatilities1) - 2 * correlations * sqrt_var_eff + tf.math.square(sqrt_var_eff))
        sqrt_var = sqrt_var_ * tf.math.sqrt(expiries)

        #if not is_normal_volatility:  # lognormal model # Don't need to differentiate between modes 
        d1 = tf.math.divide_no_nan(tf.math.log(forwards1 / (forwards2 + strikes)), sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var
        #d1 = tf.math.divide_no_nan(tf.math.log(forwards / strikes),#ln(S/K)
        #                         sqrt_var) + sqrt_var / 2
        #d2 = d1 - sqrt_var
        undiscounted_calls = tf.where(sqrt_var > 0,
                                        forwards1 * _ncdf(d1) - (forwards2 + strikes) * _ncdf(d2),
                                        tf.math.maximum(forwards1 - forwards2 - strikes, 0.0))#TODO
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        
        #Wikipedia: For spread put options, P = max(0, K - S_1 + S_2)
        #PUT: strikes*_ncdf(-d2) - forwards*_ncdf(-d1)
        undiscounted_puts = tf.where(sqrt_var > 0, 
                                    (forwards2 + strikes) * _ncdf(-d2) - forwards1 * _ncdf(-d1), 
                                    tf.math.maximum(forwards2 + strikes - forwards1, 0.0))

        return discount_factors * undiscounted_puts


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
