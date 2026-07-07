# LLM recommendation explanations - generated examples

Model: `claude-opus-4-8`. Each pairing lists the item context handed to the model and its top SHAP signals (feature=value, signed contribution), so every blurb can be checked against exactly what the prompt contained (Step 9 verification evidence).

## Preserved: before the value-aware gloss fix

The first generation run glossed one-hot signals by name and ignored their value. For the unknown visitor (route `global_popularity`, item `bb50f2e236e5...`), whose region one-hots are all 0, the prompt asserted "the shopper is from the Northeast region" and the blurb came back:

> We're showing you this well-photographed health & beauty product because its detailed listing, price, and popularity with shoppers in your Northeast region make it a strong match for you.

That region claim is invented - the system doesn't know the visitor's region. `test_signal_gloss_is_value_aware` pins the fix; unknown-region visitors now get a 'region is unknown' gloss instead of a negated one.

## route: two_stage

**66a0a6e02b7f… / cool_stuff / R$ 119.99**
- context given to the model: review 4.52/5, bought by 32 customers in the window
- signals: p_popularity=32 (+1.65), category_match=1 (-0.79), distance_km=328.4 (+0.43), region_Southeast=1 (+0.35), p_review_mean=4.516 (+0.28)
- explanation (live): A crowd-pleaser in the cool_stuff category you already love, this well-reviewed pick is popular with fellow shoppers and ships from a seller close to you in the Southeast.

**6e1b14d3cbb5… / cool_stuff / R$ 229.99**
- context given to the model: review 4.56/5, bought by 16 customers in the window
- signals: p_popularity=16 (+1.74), cf_signal=0.25 (+0.58), s_seller_order_count=924 (-0.46), p_product_weight_g=1625 (-0.36), distance_km=328.4 (+0.36)
- explanation (live): Because you keep coming back for cool_stuff, we picked this popular, well-reviewed pick that shoppers like you often buy—and it ships from a seller close to you.

**db272366bfa8… / health_beauty / R$ 79.90**
- context given to the model: review 4.14/5, bought by 0 customers in the window
- signals: p_product_weight_g=330 (+0.39), c_frequency=2 (+0.32), c_monetary=317.9 (-0.24), c_recency_days=45 (-0.19), p_review_mean=4.144 (+0.17)
- explanation (live): Because you're a returning shopper and this well-rated health & beauty pick has earned solid reviews from others, we thought it might be a nice fit for you.

## route: regional_popularity

**aca2eb7d00ea… / furniture_decor / R$ 69.90**
- context given to the model: review 4.15/5, bought by 221 customers in the window
- signals: p_popularity=221 (-0.82), p_product_photos_qty=6 (+0.35), s_seller_order_count=493 (+0.32), s_seller_review_mean=4.262 (+0.26), p_category_satisfaction_rate=0.7479 (-0.18)
- explanation (live): You're seeing this well-photographed home décor piece because it comes from a trusted, high-volume seller with strong buyer ratings.

**53b36df67ebb… / watches_gifts / R$ 154.45**
- context given to the model: review 3.89/5, bought by 66 customers in the window
- signals: p_popularity=66 (-0.67), s_seller_order_count=567 (+0.46), p_price_band=4 (+0.22), region_Southeast=1 (-0.14), p_freight_ratio=0.1118 (+0.14)
- explanation (live): This watch comes from a top-selling seller, sits in a popular price tier, and offers shipping that's affordable relative to its price—a solid pick for you.

**3dd2a17168ec… / computers_accessories / R$ 149.90**
- context given to the model: review 4.21/5, bought by 86 customers in the window
- signals: p_popularity=86 (-1.12), p_median_price_w=149.9 (+0.15), p_price_band=4 (+0.15), s_seller_review_mean=4.164 (+0.13), distance_km=2114 (-0.12)
- explanation (live): You're seeing this well-priced computer accessory because it's sold by a highly rated seller and offers great value in its price range.

## route: global_popularity

**bb50f2e236e5… / health_beauty / R$ 325.00**
- context given to the model: review 4.2/5, bought by 69 customers in the window
- signals: p_product_photos_qty=6 (+0.78), p_price_band=4 (+0.66), p_product_description_lenght=2644 (+0.54), p_median_price_w=325 (+0.41), region_Northeast=0 (+0.29)
- explanation (live): With clear photos and a detailed description behind it, this well-reviewed health & beauty pick has already been chosen by dozens of shoppers, so we thought you'd want a closer look.

**aca2eb7d00ea… / furniture_decor / R$ 69.90**
- context given to the model: review 4.15/5, bought by 221 customers in the window
- signals: p_popularity=221 (-0.72), s_seller_order_count=493 (+0.40), p_product_photos_qty=6 (+0.28), s_seller_review_mean=4.262 (+0.20), p_price_band=1 (-0.15)
- explanation (live): From a top-selling, highly rated seller, this affordable furniture and decor piece comes with plenty of photos so you can see exactly what you're getting.

**3dd2a17168ec… / computers_accessories / R$ 149.90**
- context given to the model: review 4.21/5, bought by 86 customers in the window
- signals: p_popularity=86 (-0.94), p_median_price_w=149.9 (+0.35), s_seller_review_mean=4.164 (+0.17), region_Southeast=0 (+0.17), p_price_band=4 (+0.14)
- explanation (live): This computer accessory is well-priced at R$149.90, comes from a highly rated seller, and has earned solid reviews from dozens of shoppers—a reliable pick to start with.
