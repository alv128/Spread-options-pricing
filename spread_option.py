# Copyright 2019 Google LLC
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

"""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.black_scholes import vanilla_prices


def spread_option_price(volatilities,
                        correlations,
                        strikes,
                        expiries,
                        spots,
                        forwards,
                        discount_rates,
                        dividend_rates,
                        discount_factors,
                        #sigma_const,
                        is_call_options,
                        is_normal_volatility: bool = False,
                        dtype=None,
                        name=None):

    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')

    #Kirk's approximation using WKB method for the two-dimensional Black-Scholes models for spread-options
    with tf.name_scope(name or 'spread_option_price'):

        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        #strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strike_price')#2xn tensor
        dtype = strikes.dtype
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')#2xn tensor
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        correlations = tf.convert_to_tensor(correlations, dtype=dtype, name='correlations')#1x1
        
        # dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype, name='dividend_rates')#1xn
        # spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')#2xn tensor, for the two assets 

        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
            discount_factors = tf.exp(-discount_rates * expiries)
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = -tf.math.log(discount_factors) / expiries
        else:
            discount_rates = tf.convert_to_tensor(0.0, dtype=dtype, name='discount_rates')
            discount_factors = tf.convert_to_tensor(1.0, dtype=dtype, name='discount_factors')

        if dividend_rates is None:
            dividend_rates = tf.convert_to_tensor(
                0.0, dtype=dtype, name='dividend_rates')

        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
            spots = forwards * tf.exp(-(discount_rates - dividend_rates) * expiries)
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)


        #calculate sigma_eff, if constant for sigma in Kirk's approximation is provided
        spot1 = spots[0][0]
        spot2 = spots[1][0]
        var2 = volatilities[1][0]
        var1 = volatilities[0][0]
        sigma_eff = var2 * (spot2/(spot2 + strikes[0][0]*tf.exp(-dividend_rates[0]*expiries[0])))

        #if sigma_const==None:# use WKB approximation
        sigma_ = tf.math.sqrt(var1 - 2*correlations[0]*tf.math.sqrt(var1)*tf.math.sqrt(sigma_eff) + sigma_eff)
        #else:
        #    sigma_ = tf.math.sqrt(var1 - 2*correlations[0]*tf.math.sqrt(var1)*tf.math.sqrt(sigma_const) + sigma_const)

        d1 = tf.math.log(spot1) - tf.math.log(spot2 + strikes[0][0]*tf.exp(-dividend_rates[0]*expiries[0]))
        d1 = d1/(tf.math.sqrt(sigma_*expiries[0])) + 0.5*tf.math.sqrt(sigma_*expiries[0])
        d2 = d1 + tf.math.sqrt(sigma_*expiries[0])

        #regular black scholes model for option price
        #C=SN(d1) - Kexp(-rt)N(d2)
        #use regular option pricing with above variables as input parameters

        return vanilla_prices.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        #forwards=asian_forwards,
        dividend_rates=dividend_rates,
        #discount_factors=discount_factors,
        is_call_options=is_call_options,
        dtype=dtype)