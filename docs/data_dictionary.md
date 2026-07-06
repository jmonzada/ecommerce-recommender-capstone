# Olist Brazilian eCommerce Data Dictionary

> **Provenance:** drafted by an LLM (Claude, `claude-opus-4-8`) from schema profiles
> computed directly from the raw CSVs, then human-verified against the data
> (2026-07-06). The generation script is `src/llm_docs.py`; the exact prompt and the
> unedited draft are preserved in `docs/llm/` — diff them against this file to see
> the corrections. Verified corrections: `review_id` is **not** unique (the draft
> claimed it was), plus the notes marked *(verified)* below.

The Olist dataset is a relational collection of anonymized records from a Brazilian marketplace connecting small merchants (sellers) to customers across Brazil. The tables are linked by shared keys: `orders` is the central fact table (keyed by `order_id`), joining to `customers` via `customer_id`, to `order_items`, `order_payments`, and `order_reviews` via `order_id`, and through order items to `products` (`product_id`) and `sellers` (`seller_id`). Products can be translated to English categories via `product_category_name_translation`, and both customers and sellers link to `geolocation` through zip code prefixes. Together these tables allow reconstruction of the full lifecycle of a purchase from order placement through delivery, payment, and review.

## olist_customers_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| customer_id | object | 0.0 | Per-order customer key that uniquely identifies a customer within a single order and links to the orders table. | Hash string |
| customer_unique_id | object | 0.0 | Stable identifier for the underlying person, allowing repeat purchases across multiple orders to be linked. | Hash string |
| customer_zip_code_prefix | int64 | 0.0 | First digits of the customer's postal code, used to join to geolocation. | Brazilian CEP prefix |
| customer_city | object | 0.0 | Name of the customer's city. | Lowercase city name |
| customer_state | object | 0.0 | Two-letter code of the customer's Brazilian state. | 27 state codes (e.g. SP, SC, MG, PR) |

## olist_geolocation_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| geolocation_zip_code_prefix | int64 | 0.0 | First digits of a postal code used to associate coordinates with customers and sellers. | Brazilian CEP prefix |
| geolocation_lat | float64 | 0.0 | Latitude coordinate associated with the zip code prefix. | Decimal degrees |
| geolocation_lng | float64 | 0.0 | Longitude coordinate associated with the zip code prefix. | Decimal degrees |
| geolocation_city | object | 0.0 | Name of the city for the geolocation point. | Lowercase city name |
| geolocation_state | object | 0.0 | Two-letter code of the Brazilian state for the geolocation point. | 27 state codes (e.g. SP, RN, AC, RJ) |

## olist_order_items_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| order_id | object | 0.0 | Identifier of the order to which this item belongs, linking to the orders table. | Hash string |
| order_item_id | int64 | 0.0 | Sequential number distinguishing each item line within a single order. | Integer 1–21 |
| product_id | object | 0.0 | Identifier of the purchased product, linking to the products table. | Hash string |
| seller_id | object | 0.0 | Identifier of the seller fulfilling this item, linking to the sellers table. | Hash string |
| shipping_limit_date | object | 0.0 | Deadline by which the seller must hand the item to the logistics partner. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| price | float64 | 0.0 | Item price charged to the customer, excluding freight. | BRL |
| freight_value | float64 | 0.0 | Freight (shipping) cost allocated to this item. | BRL |

## olist_order_payments_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| order_id | object | 0.0 | Identifier of the order being paid, linking to the orders table. | Hash string |
| payment_sequential | int64 | 0.0 | Sequence number when an order is paid using multiple payment methods. | Integer 1–29 |
| payment_type | object | 0.0 | Method used for this payment. | credit_card, boleto, voucher, debit_card, not_defined |
| payment_installments | int64 | 0.0 | Number of installments chosen by the customer for this payment. | Integer 0–24 |
| payment_value | float64 | 0.0 | Monetary amount of this payment record. | BRL |

## olist_order_reviews_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| review_id | object | 0.0 | Identifier of the review submitted by the customer. **Not unique** *(verified: 814 duplicated values, and 547 orders carry more than one review — deduplicate before joining)*. | Hash string |
| order_id | object | 0.0 | Identifier of the reviewed order, linking to the orders table. | Hash string |
| review_score | int64 | 0.0 | Customer satisfaction rating for the order. | Integer 1–5 |
| review_comment_title | object | 88.3 | Optional free-text title of the customer's review, in Portuguese. | Free text |
| review_comment_message | object | 58.7 | Optional free-text body of the customer's review, in Portuguese. | Free text |
| review_creation_date | object | 0.0 | Date the review survey was sent to the customer. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| review_answer_timestamp | object | 0.0 | Timestamp when the customer submitted their review answer. | Timestamp (YYYY-MM-DD HH:MM:SS) |

## olist_orders_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| order_id | object | 0.0 | Unique identifier of the order, the central key across the dataset. | Hash string |
| customer_id | object | 0.0 | Per-order customer key linking the order to the customers table. | Hash string |
| order_status | object | 0.0 | Current lifecycle status of the order. | delivered, invoiced, shipped, processing, canceled, unavailable, approved, created |
| order_purchase_timestamp | object | 0.0 | Timestamp when the customer placed the order. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| order_approved_at | object | 0.2 | Timestamp when the payment was approved. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| order_delivered_carrier_date | object | 1.8 | Timestamp when the order was handed to the logistics carrier. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| order_delivered_customer_date | object | 3.0 | Timestamp when the order was delivered to the customer. | Timestamp (YYYY-MM-DD HH:MM:SS) |
| order_estimated_delivery_date | object | 0.0 | Delivery date estimate given to the customer at purchase. | Timestamp (YYYY-MM-DD HH:MM:SS) |

## olist_products_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| product_id | object | 0.0 | Unique identifier of the product. | Hash string |
| product_category_name | object | 1.9 | Product category label in Portuguese, joinable to the English translation table. | Portuguese category name |
| product_name_lenght | float64 | 1.9 | Character count of the product name — column name misspelled in the source data *(sic)*. | Number of characters |
| product_description_lenght | float64 | 1.9 | Character count of the product description — column name misspelled in the source data *(sic)*. | Number of characters |
| product_photos_qty | float64 | 1.9 | Number of photos published for the product. | Count |
| product_weight_g | float64 | 0.0 | Weight of the product *(verified: 2 products have null weight and dimensions — the 0.0% is rounding)*. | Grams |
| product_length_cm | float64 | 0.0 | Length of the product package. | cm |
| product_height_cm | float64 | 0.0 | Height of the product package. | cm |
| product_width_cm | float64 | 0.0 | Width of the product package. | cm |

## olist_sellers_dataset.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| seller_id | object | 0.0 | Unique identifier of the seller. | Hash string |
| seller_zip_code_prefix | int64 | 0.0 | First digits of the seller's postal code, used to join to geolocation. | Brazilian CEP prefix |
| seller_city | object | 0.0 | Name of the seller's city. | Lowercase city name |
| seller_state | object | 0.0 | Two-letter code of the seller's Brazilian state. | 23 state codes (e.g. SP, RJ, PE, PR) |

## product_category_name_translation.csv

| Column | Type | Missing % | Description | Units / allowed values |
|---|---|---|---|---|
| product_category_name | object | 0.0 | Product category label in Portuguese, matching the products table. | Portuguese category name |
| product_category_name_english | object | 0.0 | English translation of the product category label. | English category name |

---

**Note:** All timestamp and date columns are recorded in local Brazilian time.