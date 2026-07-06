You are documenting the Olist Brazilian eCommerce dataset
(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for a machine
learning project's data dictionary.

Below are schema profiles for each raw CSV, computed directly from the data.
Write a complete data dictionary in markdown: one section per file, with a table
containing exactly these columns: Column | Type | Missing % | Description | Units / allowed values.

Rules:
- Copy Type and Missing % verbatim from the profiles - do not restate or round them.
- Descriptions must be concise (one sentence) and specific to Olist's marketplace
  context (e.g. customer_id is per-order; customer_unique_id identifies the person).
- Monetary columns are in Brazilian reais (BRL); dimensions in cm; weight in grams.
- For any column where you are not confident about the meaning, append "(verify)"
  to the description rather than guessing confidently.
- Start with a short intro paragraph explaining the dataset's relational structure,
  and end with a note that timestamps are local Brazilian time.
- Output only the markdown document, no preamble.

### olist_customers_dataset.csv  (99441 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| customer_id | object | 0.0 | 99441 | 06b8999e2fba1a1fbc88172c00ba8bc7, 18955e83d337fd6b2def6b18a428ac77, 4e7b3e00288586ebd08712fdd0374a03, b2b6027bc5c5109e529d4dc6358b12c3 |
| customer_unique_id | object | 0.0 | 96096 | 861eff4711a542e4b93843c6dd7febb0, 290c77bc529b7ac935b93aa66c333dc3, 060e732b5b29e8181a18229c7b0b2b5e, 259dac757896d24d7702b9acbbff3f3c |
| customer_zip_code_prefix | int64 | 0.0 | 14994 | min 1003, max 99990, median 24416 |
| customer_city | object | 0.0 | 4119 | franca, sao bernardo do campo, sao paulo, mogi das cruzes |
| customer_state | object | 0.0 | 27 | SP, SC, MG, PR |

### olist_geolocation_dataset.csv  (1000163 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| geolocation_zip_code_prefix | int64 | 0.0 | 19015 | min 1001, max 99990, median 26530 |
| geolocation_lat | float64 | 0.0 | 717360 | min -36.6054, max 45.0659, median -22.9194 |
| geolocation_lng | float64 | 0.0 | 717613 | min -101.467, max 121.105, median -46.6379 |
| geolocation_city | object | 0.0 | 8011 | sao paulo, são paulo, sao bernardo do campo, jundiaí |
| geolocation_state | object | 0.0 | 27 | SP, RN, AC, RJ |

### olist_order_items_dataset.csv  (112650 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| order_id | object | 0.0 | 98666 | 00010242fe8c5a6d1ba2dd792cb16214, 00018f77f2f0320c557190d7a144bdd3, 000229ec398224ef6ca0657da4fc703e, 00024acbcdf0a6daa1e931b038114c75 |
| order_item_id | int64 | 0.0 | 21 | min 1, max 21, median 1 |
| product_id | object | 0.0 | 32951 | 4244733e06e7ecb4970a6e2683c13e61, e5f2d52b802189ee658865ca93d83a8f, c777355d18b72b67abbeef9df44fd0fd, 7634da152a4610f1595efa32f14722fc |
| seller_id | object | 0.0 | 3095 | 48436dade18ac8b2bce089ec2a041202, dd7ddc04e1b6c2c614352b383efe2d36, 5b51032eddd242adc84c38acab88f23d, 9d7a1d34a5052409006425275ba1c2b4 |
| shipping_limit_date | object | 0.0 | 93318 | 2017-09-19 09:45:35, 2017-05-03 11:05:13, 2018-01-18 14:48:30, 2018-08-15 10:10:18 |
| price | float64 | 0.0 | 5968 | min 0.85, max 6735, median 74.99 |
| freight_value | float64 | 0.0 | 6999 | min 0, max 409.68, median 16.26 |

### olist_order_payments_dataset.csv  (103886 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| order_id | object | 0.0 | 99440 | b81ef226f3fe1789b1e8b2acac839d17, a9810da82917af2d9aefd1278f1dcfa0, 25e8ea4e93396b6fa0d3dd708e76c1bd, ba78997921bbcdc1373bb41e913ab953 |
| payment_sequential | int64 | 0.0 | 29 | min 1, max 29, median 1 |
| payment_type | object | 0.0 | 5 | credit_card, boleto, voucher, debit_card |
| payment_installments | int64 | 0.0 | 24 | min 0, max 24, median 1 |
| payment_value | float64 | 0.0 | 29077 | min 0, max 13664.1, median 100 |

### olist_order_reviews_dataset.csv  (99224 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| review_id | object | 0.0 | 98410 | 7bc2406110b926393aa56f80a40eba40, 80e641a11e56f04c1ad469d5645fdfde, 228ce5500dc1d8e020d8d1322874b6f0, e64fb393e7b32834bb789ff8bb30750e |
| order_id | object | 0.0 | 98673 | 73fc7af87114b39712e6da79b0a377eb, a548910a1c6147796b98fdf73dbeba33, f9e4b658b201a9f2ecdecbb34bed034b, 658677c97b385a9be170737859d3511b |
| review_score | int64 | 0.0 | 5 | min 1, max 5, median 5 |
| review_comment_title | object | 88.3 | 4527 | (free text - examples omitted) |
| review_comment_message | object | 58.7 | 36159 | (free text - examples omitted) |
| review_creation_date | object | 0.0 | 636 | 2018-01-18 00:00:00, 2018-03-10 00:00:00, 2018-02-17 00:00:00, 2017-04-21 00:00:00 |
| review_answer_timestamp | object | 0.0 | 98248 | 2018-01-18 21:46:59, 2018-03-11 03:05:13, 2018-02-18 14:36:24, 2017-04-21 22:02:06 |

### olist_orders_dataset.csv  (99441 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| order_id | object | 0.0 | 99441 | e481f51cbdc54678b7cc49136f2d6af7, 53cdb2fc8bc7dce0b6741e2150273451, 47770eb9100c2d0c44946d9cf07ec65d, 949d5b44dbf5de918fe9c16f97b45f8a |
| customer_id | object | 0.0 | 99441 | 9ef432eb6251297304e76186b10a928d, b0830fb4747a6c6d20dea0b8c802d7ef, 41ce2a54c0b03bf3443c3d931a367089, f88197465ea7920adcdbec7375364d82 |
| order_status | object | 0.0 | 8 | delivered, invoiced, shipped, processing |
| order_purchase_timestamp | object | 0.0 | 98875 | 2017-10-02 10:56:33, 2018-07-24 20:41:37, 2018-08-08 08:38:49, 2017-11-18 19:28:06 |
| order_approved_at | object | 0.2 | 90733 | 2017-10-02 11:07:15, 2018-07-26 03:24:27, 2018-08-08 08:55:23, 2017-11-18 19:45:59 |
| order_delivered_carrier_date | object | 1.8 | 81018 | 2017-10-04 19:55:00, 2018-07-26 14:31:00, 2018-08-08 13:50:00, 2017-11-22 13:39:59 |
| order_delivered_customer_date | object | 3.0 | 95664 | 2017-10-10 21:25:13, 2018-08-07 15:27:45, 2018-08-17 18:06:29, 2017-12-02 00:28:42 |
| order_estimated_delivery_date | object | 0.0 | 459 | 2017-10-18 00:00:00, 2018-08-13 00:00:00, 2018-09-04 00:00:00, 2017-12-15 00:00:00 |

### olist_products_dataset.csv  (32951 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| product_id | object | 0.0 | 32951 | 1e9e8ef04dbcff4541ed26657ea517e5, 3aa071139cb16b67ca9e5dea641aaa2f, 96bd76ec8810374ed1b65e291975717f, cef67bcfe19066a932b7673e239eb23d |
| product_category_name | object | 1.9 | 73 | perfumaria, artes, esporte_lazer, bebes |
| product_name_lenght | float64 | 1.9 | 66 | min 5, max 76, median 51 |
| product_description_lenght | float64 | 1.9 | 2960 | min 4, max 3992, median 595 |
| product_photos_qty | float64 | 1.9 | 19 | min 1, max 20, median 1 |
| product_weight_g | float64 | 0.0 | 2204 | min 0, max 40425, median 700 |
| product_length_cm | float64 | 0.0 | 99 | min 7, max 105, median 25 |
| product_height_cm | float64 | 0.0 | 102 | min 2, max 105, median 13 |
| product_width_cm | float64 | 0.0 | 95 | min 6, max 118, median 20 |

### olist_sellers_dataset.csv  (3095 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| seller_id | object | 0.0 | 3095 | 3442f8959a84dea7ee197c632cb2df15, d1b65fc7debc3361ea86b5f14c68d2e2, ce3ad9de960102d0677a81f5d0bb7b2d, c0f3eea2e14555b6faeea3dd58c1b1c3 |
| seller_zip_code_prefix | int64 | 0.0 | 2246 | min 1001, max 99730, median 14940 |
| seller_city | object | 0.0 | 611 | campinas, mogi guacu, rio de janeiro, sao paulo |
| seller_state | object | 0.0 | 23 | SP, RJ, PE, PR |

### product_category_name_translation.csv  (71 rows)

| column | dtype | missing % | n_unique | range / examples |
|---|---|---|---|---|
| product_category_name | object | 0.0 | 71 | beleza_saude, informatica_acessorios, automotivo, cama_mesa_banho |
| product_category_name_english | object | 0.0 | 71 | health_beauty, computers_accessories, auto, bed_bath_table |
