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

#spread option price: P_0(S_1, S_2, t) = S_1*N(d_1) - (S_2 + K*e^(-rt))*N(d_2)
#S = D*F, F = S/D forward price, D = e^(-rt)
#regular option price:       P_0(S, t) = S*N(d_1) - K*e^(-rt)*N(d_2)

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
                        dividend_rates,#if provided, substract from discount_rates, discount_rates - dividend_rates
                        discount_factors,#e^(-rt), mutually exclusive to discount_rates
                        #sigma_const,
                        is_call_options,#if not provided, assume call options
                        is_normal_volatility: bool = False,
                        dtype=None,
                        name=None):

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities1 = tf.convert_to_tensor(
        volatilities1, dtype=dtype, name='volatilities1')
    volatilities2 = tf.convert_to_tensor(
        volatilities2, dtype=dtype, name='volatilities2')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

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
    if dividend_rates is None:
      dividend_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='dividend_rates')

    
    #if forwards is not None:
    #    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    #else:
    #    spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
    #    forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)

    if (forwards1 is not None) == (forwards2 is not None):
        forwards1 = tf.convert_to_tensor(forwards1, dtype=dtype, name='forwards1')
        forwards2 = tf.convert_to_tensor(forwards2, dtype=dtype, name='forwards2')
    else:
        spots1 = tf.convert_to_tensor(spots1, dtype=dtype, name='spots1')
        spots2 = tf.convert_to_tensor(spots2, dtype=dtype, name='spots2')
        forwards1 = spots1 * tf.exp((discount_rates - dividend_rates) * expiries)
        forwards2 = spots2 * tf.exp((discount_rates - dividend_rates) * expiries)
        
    sqrt_var1 = volatilities1 * tf.math.sqrt(expiries)#sigma_s * sqrt(t)
    sqrt_var2 = volatilities2 * tf.math.sqrt(expiries)
    if not is_normal_volatility:  # lognormal model
        d1 = tf.math.divide_no_nan(tf.math.log(forwards / strikes),#ln(S/K)
                                 sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var
        undiscounted_calls = tf.where(sqrt_var > 0,
                                        forwards * _ncdf(d1) - strikes * _ncdf(d2),
                                        tf.math.maximum(forwards - strikes, 0.0))
    else:  # normal model
        d1 = tf.math.divide_no_nan((forwards - strikes), sqrt_var)
        undiscounted_calls = tf.where(
            sqrt_var > 0.0, (forwards - strikes) * _ncdf(d1) +
            sqrt_var * tf.math.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi),
            tf.math.maximum(forwards - strikes, 0.0))





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