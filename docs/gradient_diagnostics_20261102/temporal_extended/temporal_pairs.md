# B1/B2 temporal gradient pairs

Status: **complete**; 12/12 valid window × seed pairs.

Each experiment has three source windows × two noise seeds. The six observations share training checkpoints and are not independent training repeats. No significance inference is made.

Δ = step 1800 minus step 1200. Positive auxiliary effect or cancellation means local resistance to descent. Auxiliary effect is defined only when main-only descent is negative.

Percentage metric Δ uses percentage_points; cosine Δ uses absolute cosine units (a signed difference, not |Δ|). No window-averaged cosine is presented as a 384-source gradient cosine.

Improved/worsened labels mean only more same-direction alignment, less local cancellation/resistance, or fewer ascent cases. They do not indicate evaluation improvement. Other metrics show increase/decrease without a quality direction.

## exp32_3maps_B1_S3_J8

Valid pairs: 6/6.

| Metric | Δ unit | Median Δ | Range Δ | Improved / worsened |
|---|---|---:|---:|---:|
| aux_loc_raw_norm (n=6) | native_units | -0.0013755 | -0.0099257 to 0.003401 | increase 2 / decrease 4; direction not assigned |
| aux_loc_weighted_norm (n=6) | native_units | -0.0013755 | -0.0099257 to 0.003401 | increase 2 / decrease 4; direction not assigned |
| aux_over_main_pct (n=6) | percentage_points | -1.3844 | -9.5938 to 1.5452 | increase 2 / decrease 4; direction not assigned |
| gen_aux_effect_vs_main_descent_pct (n=6) | percentage_points | -0.23235 | -0.58625 to 0.46702 | 4 / 2 |
| gen_global_main_only_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_global_total_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_layer_main_only_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_layer_max_cross_main_cancellation_pct (n=6) | percentage_points | 1.2662 | -1.0395 to 5.3718 | 2 / 4 |
| gen_layer_total_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_loc_main_cosine (n=6) | absolute | -0.00017359 | -0.0044601 to 0.0032688 | 3 / 3 |
| gen_main_only_descent_first_order (n=6) | native_units | -0.00032569 | -0.0014777 to 0.001251 | increase 2 / decrease 4; direction not assigned |
| gen_other_main_cancellation_pct (n=6) | percentage_points | 0.026942 | -0.36939 to 0.3951 | 3 / 3 |
| gen_raw_norm (n=6) | native_units | 0.0020754 | -0.0067216 to 0.0072241 | increase 4 / decrease 2; direction not assigned |
| gen_total_descent_first_order (n=6) | native_units | -0.00032269 | -0.0015358 to 0.001235 | increase 2 / decrease 4; direction not assigned |
| gen_weighted_norm (n=6) | native_units | 0.0020754 | -0.0067216 to 0.0072241 | increase 4 / decrease 2; direction not assigned |
| loc_aux_effect_vs_main_descent_pct (n=6) | percentage_points | 0.21764 | -0.13952 to 0.99327 | 1 / 5 |
| loc_global_main_only_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_global_total_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_layer_main_only_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_layer_max_cross_main_cancellation_pct (n=6) | percentage_points | -0.10364 | -1.1026 to 2.3453 | 4 / 2 |
| loc_layer_total_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_main_only_descent_first_order (n=6) | native_units | -5.1413e-05 | -0.00082414 to 5.6774e-05 | increase 1 / decrease 5; direction not assigned |
| loc_other_main_cancellation_pct (n=6) | percentage_points | 0.042872 | -0.28886 to 0.5033 | 2 / 4 |
| loc_raw_norm (n=6) | native_units | 0.00029087 | -0.00030728 to 0.00316 | increase 5 / decrease 1; direction not assigned |
| loc_total_descent_first_order (n=6) | native_units | -4.2446e-05 | -0.00082296 to 5.5222e-05 | increase 1 / decrease 5; direction not assigned |
| loc_weighted_norm (n=6) | native_units | 0.002327 | -0.0024582 to 0.02528 | increase 5 / decrease 1; direction not assigned |
| p_base_dot (n=6) | native_units | 1.6831e-07 | -2.0924e-06 to 6.1485e-06 | increase 4 / decrease 2; direction not assigned |
| p_cosine_to_total (n=6) | absolute | 0.0005994 | -0.019913 to 0.039698 | 3 / 3 |
| p_cross_aux_cancellation_pct (n=5) | percentage_points | 7.0918 | -12.622 to 40.166 | 1 / 4 |
| p_other_aux_dot (n=6) | native_units | -2.9642e-07 | -6.2556e-07 to 4.2556e-07 | increase 1 / decrease 5; direction not assigned |
| p_total_dot (n=6) | native_units | 1.9031e-07 | -2.6143e-06 to 5.523e-06 | increase 3 / decrease 3; direction not assigned |
| perception_aux_loc_cosine (n=6) | absolute | -0.014892 | -0.046437 to 0.012216 | increase 1 / decrease 5; direction not assigned |
| perception_raw_norm (n=6) | native_units | 0.00072929 | -0.0011896 to 0.0029071 | increase 3 / decrease 3; direction not assigned |
| perception_weighted_norm (n=6) | native_units | 3.6464e-05 | -5.948e-05 to 0.00014535 | increase 3 / decrease 3; direction not assigned |
| u_base_dot (n=6) | native_units | 5.2247e-06 | -0.00079612 to 0.000225 | increase 4 / decrease 2; direction not assigned |
| u_cosine_to_total (n=6) | absolute | 0.00021799 | -0.085677 to 0.011217 | 3 / 3 |
| u_cross_aux_cancellation_pct (n=6) | percentage_points | 0.028436 | -0.017493 to 0.30469 | 1 / 5 |
| u_other_aux_dot (n=6) | native_units | -2.9642e-07 | -6.2556e-07 to 4.2556e-07 | increase 1 / decrease 5; direction not assigned |
| u_total_dot (n=6) | native_units | 4.8581e-06 | -0.0007965 to 0.00022543 | increase 4 / decrease 2; direction not assigned |

### Window sign consistency

A window is same direction when both seed deltas have the same nonzero sign. Shared direction counts the largest group of windows with the same such sign; pending and undefined windows are excluded.

| Metric | Offset | Δ seed 17 | Δ seed 29 | Classification | Same direction windows | Shared direction windows |
|---|---:|---:|---:|---|---:|---:|
| gen_loc_main_cosine (absolute) | 0 | -0.0044601 | 0.0006139 | mixed | 2 | 1 |
| gen_loc_main_cosine (absolute) | 128 | 0.00039319 | 0.0032688 | both_positive | 2 | 1 |
| gen_loc_main_cosine (absolute) | 256 | -0.0019411 | -0.00074036 | both_negative | 2 | 1 |
| perception_aux_loc_cosine (absolute) | 0 | -0.012366 | -0.046437 | both_negative | 2 | 2 |
| perception_aux_loc_cosine (absolute) | 128 | -0.017419 | 0.012216 | mixed | 2 | 2 |
| perception_aux_loc_cosine (absolute) | 256 | -0.022005 | -0.0047103 | both_negative | 2 | 2 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 0 | -0.31663 | -0.58625 | both_negative | 1 | 1 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 128 | -0.2599 | 0.010255 | mixed | 1 | 1 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 256 | 0.46702 | -0.2048 | mixed | 1 | 1 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 0 | 0.99327 | 0.096326 | both_positive | 2 | 2 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 128 | -0.13952 | 0.71333 | mixed | 2 | 2 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 256 | 0.31906 | 0.11622 | both_positive | 2 | 2 |
| gen_other_main_cancellation_pct (percentage_points) | 0 | 0.3951 | -0.21379 | mixed | 2 | 1 |
| gen_other_main_cancellation_pct (percentage_points) | 128 | -0.036281 | -0.36939 | both_negative | 2 | 1 |
| gen_other_main_cancellation_pct (percentage_points) | 256 | 0.26914 | 0.090164 | both_positive | 2 | 1 |
| loc_other_main_cancellation_pct (percentage_points) | 0 | 0.5033 | 0.025107 | both_positive | 3 | 2 |
| loc_other_main_cancellation_pct (percentage_points) | 128 | -0.042108 | -0.28886 | both_negative | 3 | 2 |
| loc_other_main_cancellation_pct (percentage_points) | 256 | 0.13995 | 0.060636 | both_positive | 3 | 2 |
| p_cosine_to_total (absolute) | 0 | -0.014591 | -0.019913 | both_negative | 2 | 1 |
| p_cosine_to_total (absolute) | 128 | -0.0050459 | 0.0062447 | mixed | 2 | 1 |
| p_cosine_to_total (absolute) | 256 | 0.039698 | 0.031992 | both_positive | 2 | 1 |
| u_cosine_to_total (absolute) | 0 | -0.085677 | 0.0018203 | mixed | 0 | 0 |
| u_cosine_to_total (absolute) | 128 | 0.011217 | -0.0004021 | mixed | 0 | 0 |
| u_cosine_to_total (absolute) | 256 | 0.00083809 | -0.020528 | mixed | 0 | 0 |

## exp32_3maps_B2_S3_J8

Valid pairs: 6/6.

| Metric | Δ unit | Median Δ | Range Δ | Improved / worsened |
|---|---|---:|---:|---:|
| aux_loc_raw_norm (n=6) | native_units | 0.0006414 | -0.0034337 to 0.011715 | increase 4 / decrease 2; direction not assigned |
| aux_loc_weighted_norm (n=6) | native_units | 0.0006414 | -0.0034337 to 0.011715 | increase 4 / decrease 2; direction not assigned |
| aux_over_main_pct (n=6) | percentage_points | 1.0314 | -3.8898 to 6.8934 | increase 4 / decrease 2; direction not assigned |
| gen_aux_effect_vs_main_descent_pct (n=6) | percentage_points | -0.54274 | -2.4444 to 0.23572 | 5 / 1 |
| gen_global_main_only_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_global_total_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_layer_main_only_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_layer_max_cross_main_cancellation_pct (n=6) | percentage_points | -0.68667 | -4.6826 to 1.0352 | 4 / 2 |
| gen_layer_total_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| gen_loc_main_cosine (n=6) | absolute | 0.00029369 | -0.0019717 to 0.0030231 | 3 / 3 |
| gen_main_only_descent_first_order (n=6) | native_units | -0.00010861 | -0.0012481 to 0.0015067 | increase 2 / decrease 4; direction not assigned |
| gen_other_main_cancellation_pct (n=6) | percentage_points | -0.020342 | -0.39826 to 0.51212 | 3 / 3 |
| gen_raw_norm (n=6) | native_units | 0.00061572 | -0.0086053 to 0.0072708 | increase 3 / decrease 3; direction not assigned |
| gen_total_descent_first_order (n=6) | native_units | -0.00019103 | -0.0013773 to 0.0015233 | increase 2 / decrease 4; direction not assigned |
| gen_weighted_norm (n=6) | native_units | 0.00061572 | -0.0086053 to 0.0072708 | increase 3 / decrease 3; direction not assigned |
| loc_aux_effect_vs_main_descent_pct (n=6) | percentage_points | -0.038057 | -0.38807 to 1.6577 | 5 / 1 |
| loc_global_main_only_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_global_total_ascent (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_layer_main_only_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_layer_max_cross_main_cancellation_pct (n=6) | percentage_points | 0.1497 | -4.6151 to 0.65734 | 3 / 3 |
| loc_layer_total_ascent_count (n=6) | count | 0 | 0 to 0 | 0 / 0 |
| loc_main_only_descent_first_order (n=6) | native_units | 0.00054133 | -0.0026462 to 0.0018608 | increase 4 / decrease 2; direction not assigned |
| loc_other_main_cancellation_pct (n=6) | percentage_points | 0.005099 | -0.23646 to 0.15371 | 3 / 3 |
| loc_raw_norm (n=6) | native_units | -0.0018458 | -0.0051601 to 0.0064193 | increase 2 / decrease 4; direction not assigned |
| loc_total_descent_first_order (n=6) | native_units | 0.00054007 | -0.0026474 to 0.0018626 | increase 4 / decrease 2; direction not assigned |
| loc_weighted_norm (n=6) | native_units | -0.014767 | -0.041281 to 0.051355 | increase 2 / decrease 4; direction not assigned |
| p_base_dot (n=6) | native_units | -7.9187e-07 | -7.4754e-06 to 5.0104e-06 | increase 2 / decrease 4; direction not assigned |
| p_cosine_to_total (n=6) | absolute | -0.0034647 | -0.069741 to 0.045168 | 3 / 3 |
| p_cross_aux_cancellation_pct (n=5) | percentage_points | -2.1652 | -23.48 to 3.0497 | 3 / 2 |
| p_other_aux_dot (n=6) | native_units | 5.499e-08 | -6.7216e-07 to 2.5492e-06 | increase 4 / decrease 2; direction not assigned |
| p_total_dot (n=6) | native_units | -7.0349e-07 | -4.9262e-06 to 4.8797e-06 | increase 2 / decrease 4; direction not assigned |
| perception_aux_loc_cosine (n=6) | absolute | 0.0054766 | -0.054189 to 0.14329 | increase 4 / decrease 2; direction not assigned |
| perception_raw_norm (n=6) | native_units | -0.00023085 | -0.0012035 to 0.00070368 | increase 3 / decrease 3; direction not assigned |
| perception_weighted_norm (n=6) | native_units | -1.1543e-05 | -6.0174e-05 to 3.5184e-05 | increase 3 / decrease 3; direction not assigned |
| u_base_dot (n=6) | native_units | 5.7472e-05 | -3.8722e-05 to 0.00097905 | increase 4 / decrease 2; direction not assigned |
| u_cosine_to_total (n=6) | absolute | 0.025431 | -0.0050693 to 0.081138 | 5 / 1 |
| u_cross_aux_cancellation_pct (n=6) | percentage_points | 0.0046098 | -0.3903 to 0.13525 | 2 / 4 |
| u_other_aux_dot (n=6) | native_units | 5.499e-08 | -6.7216e-07 to 2.5492e-06 | increase 4 / decrease 2; direction not assigned |
| u_total_dot (n=6) | native_units | 5.7527e-05 | -3.6173e-05 to 0.00097838 | increase 4 / decrease 2; direction not assigned |

### Window sign consistency

A window is same direction when both seed deltas have the same nonzero sign. Shared direction counts the largest group of windows with the same such sign; pending and undefined windows are excluded.

| Metric | Offset | Δ seed 17 | Δ seed 29 | Classification | Same direction windows | Shared direction windows |
|---|---:|---:|---:|---|---:|---:|
| gen_loc_main_cosine (absolute) | 0 | -0.00029787 | 0.00075914 | mixed | 0 | 0 |
| gen_loc_main_cosine (absolute) | 128 | -0.00017175 | 0.0030231 | mixed | 0 | 0 |
| gen_loc_main_cosine (absolute) | 256 | 0.0016093 | -0.0019717 | mixed | 0 | 0 |
| perception_aux_loc_cosine (absolute) | 0 | 0.14329 | 0.0081835 | both_positive | 3 | 2 |
| perception_aux_loc_cosine (absolute) | 128 | -0.054189 | -0.010001 | both_negative | 3 | 2 |
| perception_aux_loc_cosine (absolute) | 256 | 0.010425 | 0.0027697 | both_positive | 3 | 2 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 0 | -2.4444 | 0.23572 | mixed | 2 | 2 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 128 | -1.5606 | -0.060347 | both_negative | 2 | 2 |
| gen_aux_effect_vs_main_descent_pct (percentage_points) | 256 | -0.27507 | -0.8104 | both_negative | 2 | 2 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 0 | 1.6577 | -0.072301 | mixed | 2 | 2 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 128 | -0.38807 | -0.017909 | both_negative | 2 | 2 |
| loc_aux_effect_vs_main_descent_pct (percentage_points) | 256 | -0.0081844 | -0.058204 | both_negative | 2 | 2 |
| gen_other_main_cancellation_pct (percentage_points) | 0 | 0.0018288 | 0.18844 | both_positive | 2 | 1 |
| gen_other_main_cancellation_pct (percentage_points) | 128 | -0.042513 | -0.38878 | both_negative | 2 | 1 |
| gen_other_main_cancellation_pct (percentage_points) | 256 | -0.39826 | 0.51212 | mixed | 2 | 1 |
| loc_other_main_cancellation_pct (percentage_points) | 0 | 0.14162 | -0.10048 | mixed | 0 | 0 |
| loc_other_main_cancellation_pct (percentage_points) | 128 | 0.15371 | -0.23646 | mixed | 0 | 0 |
| loc_other_main_cancellation_pct (percentage_points) | 256 | -0.064893 | 0.075091 | mixed | 0 | 0 |
| p_cosine_to_total (absolute) | 0 | -0.069741 | -0.011655 | both_negative | 2 | 1 |
| p_cosine_to_total (absolute) | 128 | -0.047807 | 0.045168 | mixed | 2 | 1 |
| p_cosine_to_total (absolute) | 256 | 0.0047255 | 0.024676 | both_positive | 2 | 1 |
| u_cosine_to_total (absolute) | 0 | 0.020713 | -0.0050693 | mixed | 2 | 2 |
| u_cosine_to_total (absolute) | 128 | 0.081138 | 0.044054 | both_positive | 2 | 2 |
| u_cosine_to_total (absolute) | 256 | 0.012867 | 0.030149 | both_positive | 2 | 2 |

## Individual pairs

All metric values below are local measurements or algebra from the paired reports.

### exp32_3maps_B1_S3_J8 · offset 0 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `16ccb75fcb3fa3969c89d94d50b12ea758d601ec186e9eb2ecf774a0b76240d0` / `8313e0557fca63f3a3b83d882d7eec4c42bbac1283e8e831b759084dad38b671`.

Matched source IDs: True (128 rows; SHA256 `bfcb5e2a0a1cdd07100c38f8da1034ab2dc53918977f5550f2fd95e2dba5b02b`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.043393 | 0.033467 | -0.0099257 (native_units) |
| aux_loc_weighted_norm | 0.043393 | 0.033467 | -0.0099257 (native_units) |
| aux_over_main_pct | 37.957 | 28.364 | -9.5938 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.12044 | -0.43707 | -0.31663 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 1.968 | 4.0807 | 2.1127 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.0084004 | 0.0039403 | -0.0044601 (absolute) |
| gen_main_only_descent_first_order | -0.0072408 | -0.0076202 | -0.00037944 (native_units) |
| gen_other_main_cancellation_pct | -0.75363 | -0.35852 | 0.3951 (percentage_points) |
| gen_raw_norm | 0.084774 | 0.087138 | 0.0023639 (native_units) |
| gen_total_descent_first_order | -0.0072495 | -0.0076535 | -0.00040403 (native_units) |
| gen_weighted_norm | 0.084774 | 0.087138 | 0.0023639 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -1.004 | -0.010733 | 0.99327 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 2.2844 | 3.4302 | 1.1458 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.00072978 | -0.00078917 | -5.9386e-05 (native_units) |
| loc_other_main_cancellation_pct | -0.93636 | -0.43306 | 0.5033 (percentage_points) |
| loc_raw_norm | 0.0095067 | 0.0099107 | 0.00040398 (native_units) |
| loc_total_descent_first_order | -0.00073711 | -0.00078926 | -5.2144e-05 (native_units) |
| loc_weighted_norm | 0.076053 | 0.079285 | 0.0032318 (native_units) |
| p_base_dot | 5.3995e-06 | 5.4891e-06 | 8.9583e-08 (native_units) |
| p_cosine_to_total | 0.068222 | 0.053631 | -0.014591 (absolute) |
| p_cross_aux_cancellation_pct | -8.6154 | -1.5236 | 7.0918 (percentage_points) |
| p_other_aux_dot | 4.6519e-07 | 8.3632e-08 | -3.8156e-07 (native_units) |
| p_total_dot | 5.8647e-06 | 5.5727e-06 | -2.9197e-07 (native_units) |
| perception_aux_loc_cosine | 0.015323 | 0.0029574 | -0.012366 (absolute) |
| perception_raw_norm | 0.013992 | 0.0169 | 0.0029071 (native_units) |
| perception_weighted_norm | 0.00069962 | 0.00084498 | 0.00014535 (native_units) |
| u_base_dot | 0.0019454 | 0.0011493 | -0.00079612 (native_units) |
| u_cosine_to_total | 0.36495 | 0.27927 | -0.085677 (absolute) |
| u_cross_aux_cancellation_pct | -0.023912 | -0.007277 | 0.016635 (percentage_points) |
| u_other_aux_dot | 4.6519e-07 | 8.3632e-08 | -3.8156e-07 (native_units) |
| u_total_dot | 0.0019459 | 0.0011494 | -0.0007965 (native_units) |

### exp32_3maps_B1_S3_J8 · offset 0 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `16ccb75fcb3fa3969c89d94d50b12ea758d601ec186e9eb2ecf774a0b76240d0` / `8313e0557fca63f3a3b83d882d7eec4c42bbac1283e8e831b759084dad38b671`.

Matched source IDs: True (128 rows; SHA256 `bfcb5e2a0a1cdd07100c38f8da1034ab2dc53918977f5550f2fd95e2dba5b02b`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.013661 | 0.012241 | -0.0014198 (native_units) |
| aux_loc_weighted_norm | 0.013661 | 0.012241 | -0.0014198 (native_units) |
| aux_over_main_pct | 8.7079 | 6.7763 | -1.9317 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 0.3439 | -0.24235 | -0.58625 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 2.8248 | 3.2445 | 0.4197 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.0091681 | 0.009782 | 0.0006139 (absolute) |
| gen_main_only_descent_first_order | -0.009307 | -0.010785 | -0.0014777 (native_units) |
| gen_other_main_cancellation_pct | -1.1862 | -1.4 | -0.21379 (percentage_points) |
| gen_raw_norm | 0.095906 | 0.10313 | 0.0072241 (native_units) |
| gen_total_descent_first_order | -0.009275 | -0.010811 | -0.0015358 (native_units) |
| gen_weighted_norm | 0.095906 | 0.10313 | 0.0072241 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -0.11265 | -0.016326 | 0.096326 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 2.1687 | 2.125 | -0.043761 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0019384 | -0.0027419 | -0.00080346 (native_units) |
| loc_other_main_cancellation_pct | -0.70859 | -0.68348 | 0.025107 (percentage_points) |
| loc_raw_norm | 0.015511 | 0.01845 | 0.002939 (native_units) |
| loc_total_descent_first_order | -0.0019406 | -0.0027423 | -0.00080172 (native_units) |
| loc_weighted_norm | 0.12409 | 0.1476 | 0.023512 (native_units) |
| p_base_dot | 3.7508e-06 | 1.6584e-06 | -2.0924e-06 (native_units) |
| p_cosine_to_total | 0.03043 | 0.010517 | -0.019913 (absolute) |
| p_cross_aux_cancellation_pct | -10.544 | 7.6214 | 18.165 (percentage_points) |
| p_other_aux_dot | 3.9548e-07 | -1.2639e-07 | -5.2188e-07 (native_units) |
| p_total_dot | 4.1463e-06 | 1.532e-06 | -2.6143e-06 (native_units) |
| perception_aux_loc_cosine | 0.033575 | -0.012862 | -0.046437 (absolute) |
| perception_raw_norm | 0.017245 | 0.016055 | -0.0011896 (native_units) |
| perception_weighted_norm | 0.00086223 | 0.00080275 | -5.948e-05 (native_units) |
| u_base_dot | 0.00016908 | 0.00017855 | 9.4715e-06 (native_units) |
| u_cosine_to_total | 0.078503 | 0.080323 | 0.0018203 (absolute) |
| u_cross_aux_cancellation_pct | -0.2339 | 0.070786 | 0.30469 (percentage_points) |
| u_other_aux_dot | 3.9548e-07 | -1.2639e-07 | -5.2188e-07 (native_units) |
| u_total_dot | 0.00016948 | 0.00017843 | 8.9497e-06 (native_units) |

### exp32_3maps_B1_S3_J8 · offset 128 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `16aae72d158d320c15ec0c77a7b31da1e46863d8933badd0c79713a903df51c7` / `f953b59ac61f13269024e8e817b50420c5c749bd7f034888d4ab72de3a6d128b`.

Matched source IDs: True (128 rows; SHA256 `68d53c3eca3ef21190d12f097b68dfe796d75792c99b316a03ff17780df9a549`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.013112 | 0.011781 | -0.0013311 (native_units) |
| aux_loc_weighted_norm | 0.013112 | 0.011781 | -0.0013311 (native_units) |
| aux_over_main_pct | 10.083 | 9.2458 | -0.83722 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 0.24811 | -0.011792 | -0.2599 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 3.0431 | 2.0037 | -1.0395 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.0037305 | 0.0041237 | 0.00039319 (absolute) |
| gen_main_only_descent_first_order | -0.0081836 | -0.007935 | 0.0002486 (native_units) |
| gen_other_main_cancellation_pct | -0.38837 | -0.42465 | -0.036281 (percentage_points) |
| gen_raw_norm | 0.090288 | 0.08889 | -0.001398 (native_units) |
| gen_total_descent_first_order | -0.0081633 | -0.0079359 | 0.00022736 (native_units) |
| gen_weighted_norm | 0.090288 | 0.08889 | -0.001398 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.14968 | 0.010161 | -0.13952 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 2.3255 | 2.162 | -0.16353 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0011084 | -0.0010516 | 5.6774e-05 (native_units) |
| loc_other_main_cancellation_pct | -0.35834 | -0.40045 | -0.042108 (percentage_points) |
| loc_raw_norm | 0.011749 | 0.011442 | -0.00030728 (native_units) |
| loc_total_descent_first_order | -0.0011067 | -0.0010515 | 5.5222e-05 (native_units) |
| loc_weighted_norm | 0.093995 | 0.091537 | -0.0024582 (native_units) |
| p_base_dot | 3.4905e-06 | 2.9925e-06 | -4.9796e-07 (native_units) |
| p_cosine_to_total | 0.036153 | 0.031107 | -0.0050459 (absolute) |
| p_cross_aux_cancellation_pct | -10.793 | -5.5286 | 5.2642 (percentage_points) |
| p_other_aux_dot | 3.7672e-07 | 1.6545e-07 | -2.1128e-07 (native_units) |
| p_total_dot | 3.8672e-06 | 3.158e-06 | -7.0924e-07 (native_units) |
| perception_aux_loc_cosine | 0.035181 | 0.017762 | -0.017419 (absolute) |
| perception_raw_norm | 0.016333 | 0.015813 | -0.00052052 (native_units) |
| perception_weighted_norm | 0.00081666 | 0.00079064 | -2.6026e-05 (native_units) |
| u_base_dot | 0.00013553 | 0.0001365 | 9.7784e-07 (native_units) |
| u_cosine_to_total | 0.07913 | 0.090347 | 0.011217 (absolute) |
| u_cross_aux_cancellation_pct | -0.27797 | -0.1212 | 0.15677 (percentage_points) |
| u_other_aux_dot | 3.7672e-07 | 1.6545e-07 | -2.1128e-07 (native_units) |
| u_total_dot | 0.0001359 | 0.00013667 | 7.6656e-07 (native_units) |

### exp32_3maps_B1_S3_J8 · offset 128 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `16aae72d158d320c15ec0c77a7b31da1e46863d8933badd0c79713a903df51c7` / `f953b59ac61f13269024e8e817b50420c5c749bd7f034888d4ab72de3a6d128b`.

Matched source IDs: True (128 rows; SHA256 `68d53c3eca3ef21190d12f097b68dfe796d75792c99b316a03ff17780df9a549`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.043549 | 0.04695 | 0.003401 (native_units) |
| aux_loc_weighted_norm | 0.043549 | 0.04695 | 0.003401 (native_units) |
| aux_over_main_pct | 29.586 | 30.869 | 1.2831 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.37049 | -0.36023 | 0.010255 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 5.2305 | 4.9379 | -0.29262 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.0050918 | -0.0018229 | 0.0032688 (absolute) |
| gen_main_only_descent_first_order | -0.0097467 | -0.010873 | -0.001126 (native_units) |
| gen_other_main_cancellation_pct | 0.56311 | 0.19372 | -0.36939 (percentage_points) |
| gen_raw_norm | 0.099004 | 0.10437 | 0.005369 (native_units) |
| gen_total_descent_first_order | -0.0097828 | -0.010912 | -0.0011291 (native_units) |
| gen_weighted_norm | 0.099004 | 0.10437 | 0.005369 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -0.59402 | 0.11931 | 0.71333 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 6.0207 | 4.9181 | -1.1026 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0014916 | -0.0015351 | -4.3441e-05 (native_units) |
| loc_other_main_cancellation_pct | 0.46041 | 0.17155 | -0.28886 (percentage_points) |
| loc_raw_norm | 0.013686 | 0.013864 | 0.00017776 (native_units) |
| loc_total_descent_first_order | -0.0015005 | -0.0015333 | -3.2749e-05 (native_units) |
| loc_weighted_norm | 0.10949 | 0.11091 | 0.0014221 (native_units) |
| p_base_dot | 2.9083e-06 | 3.1553e-06 | 2.4703e-07 (native_units) |
| p_cosine_to_total | 0.027426 | 0.033671 | 0.0062447 (absolute) |
| p_cross_aux_cancellation_pct | -11.049 | -23.671 | -12.622 (percentage_points) |
| p_other_aux_dot | 3.2134e-07 | 7.4691e-07 | 4.2556e-07 (native_units) |
| p_total_dot | 3.2296e-06 | 3.9022e-06 | 6.726e-07 (native_units) |
| perception_aux_loc_cosine | 0.0096654 | 0.021881 | 0.012216 (absolute) |
| perception_raw_norm | 0.015269 | 0.014541 | -0.00072778 (native_units) |
| perception_weighted_norm | 0.00076345 | 0.00072706 | -3.6389e-05 (native_units) |
| u_base_dot | 0.0020012 | 0.0022262 | 0.000225 (native_units) |
| u_cosine_to_total | 0.29796 | 0.29756 | -0.0004021 (absolute) |
| u_cross_aux_cancellation_pct | -0.016058 | -0.033551 | -0.017493 (percentage_points) |
| u_other_aux_dot | 3.2134e-07 | 7.4691e-07 | 4.2556e-07 (native_units) |
| u_total_dot | 0.0020015 | 0.0022269 | 0.00022543 (native_units) |

### exp32_3maps_B1_S3_J8 · offset 256 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `b06569ecd46bfbfa314d610d73c73511125511841e393bfef06c27ba7b471367` / `66cc6165aa3a1d55e54d767b513b5bcce65c9665955dc045f992af87db29ecd1`.

Matched source IDs: True (128 rows; SHA256 `b54576ae2b2e1a5c8ef6875897e49243762d8df3bd26e462ed175994d98a67eb`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.038702 | 0.04129 | 0.0025881 (native_units) |
| aux_loc_weighted_norm | 0.038702 | 0.04129 | 0.0025881 (native_units) |
| aux_over_main_pct | 27.68 | 29.225 | 1.5452 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.31165 | 0.15537 | 0.46702 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 5.3493 | 10.721 | 5.3718 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.0012189 | -0.00316 | -0.0019411 (absolute) |
| gen_main_only_descent_first_order | -0.0064575 | -0.0067294 | -0.00027193 (native_units) |
| gen_other_main_cancellation_pct | 0.17365 | 0.44279 | 0.26914 (percentage_points) |
| gen_raw_norm | 0.080428 | 0.082215 | 0.0017869 (native_units) |
| gen_total_descent_first_order | -0.0064776 | -0.0067189 | -0.00024135 (native_units) |
| gen_weighted_norm | 0.080428 | 0.082215 | 0.0017869 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -0.40787 | -0.088814 | 0.31906 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 2.3179 | 4.6632 | 2.3453 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0016397 | -0.0016553 | -1.5544e-05 (native_units) |
| loc_other_main_cancellation_pct | 0.085559 | 0.22551 | 0.13995 (percentage_points) |
| loc_raw_norm | 0.014323 | 0.014401 | 7.7818e-05 (native_units) |
| loc_total_descent_first_order | -0.0016464 | -0.0016567 | -1.0327e-05 (native_units) |
| loc_weighted_norm | 0.11458 | 0.1152 | 0.00062254 (native_units) |
| p_base_dot | 1.7187e-06 | 7.8672e-06 | 6.1485e-06 (native_units) |
| p_cosine_to_total | 0.02212 | 0.061818 | 0.039698 (absolute) |
| p_cross_aux_cancellation_pct | -41.22 | -1.0535 | 40.166 (percentage_points) |
| p_other_aux_dot | 7.0844e-07 | 8.2883e-08 | -6.2556e-07 (native_units) |
| p_total_dot | 2.4271e-06 | 7.9501e-06 | 5.523e-06 (native_units) |
| perception_aux_loc_cosine | 0.024303 | 0.0022982 | -0.022005 (absolute) |
| perception_raw_norm | 0.015064 | 0.017469 | 0.0024047 (native_units) |
| perception_weighted_norm | 0.00075319 | 0.00087343 | 0.00012024 (native_units) |
| u_base_dot | 0.0015703 | 0.0016991 | 0.00012876 (native_units) |
| u_cosine_to_total | 0.27865 | 0.27949 | 0.00083809 (absolute) |
| u_cross_aux_cancellation_pct | -0.045114 | -0.0048781 | 0.040236 (percentage_points) |
| u_other_aux_dot | 7.0844e-07 | 8.2883e-08 | -6.2556e-07 (native_units) |
| u_total_dot | 0.001571 | 0.0016992 | 0.00012813 (native_units) |

### exp32_3maps_B1_S3_J8 · offset 256 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `b06569ecd46bfbfa314d610d73c73511125511841e393bfef06c27ba7b471367` / `66cc6165aa3a1d55e54d767b513b5bcce65c9665955dc045f992af87db29ecd1`.

Matched source IDs: True (128 rows; SHA256 `b54576ae2b2e1a5c8ef6875897e49243762d8df3bd26e462ed175994d98a67eb`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.01864 | 0.017062 | -0.0015774 (native_units) |
| aux_loc_weighted_norm | 0.01864 | 0.017062 | -0.0015774 (native_units) |
| aux_over_main_pct | 12.283 | 10.137 | -2.1463 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.018497 | -0.2233 | -0.2048 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 5.8465 | 8.791 | 2.9445 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.00076615 | 2.5788e-05 | -0.00074036 (absolute) |
| gen_main_only_descent_first_order | -0.0091867 | -0.0079357 | 0.001251 (native_units) |
| gen_other_main_cancellation_pct | -0.09431 | -0.0041457 | 0.090164 (percentage_points) |
| gen_raw_norm | 0.095802 | 0.089081 | -0.0067216 (native_units) |
| gen_total_descent_first_order | -0.0091884 | -0.0079534 | 0.001235 (native_units) |
| gen_weighted_norm | 0.095802 | 0.089081 | -0.0067216 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -0.21874 | -0.10253 | 0.11622 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 4.8988 | 4.4593 | -0.43946 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0017395 | -0.0025636 | -0.00082414 (native_units) |
| loc_other_main_cancellation_pct | -0.062241 | -0.0016041 | 0.060636 (percentage_points) |
| loc_raw_norm | 0.014741 | 0.017901 | 0.00316 (native_units) |
| loc_total_descent_first_order | -0.0017433 | -0.0025662 | -0.00082296 (native_units) |
| loc_weighted_norm | 0.11793 | 0.14321 | 0.02528 (native_units) |
| p_base_dot | -3.5276e-07 | 4.4411e-06 | 4.7938e-06 (native_units) |
| p_cosine_to_total | -0.0007899 | 0.031202 | 0.031992 (absolute) |
| p_cross_aux_cancellation_pct | — | -4.4268 | — (percentage_points) |
| p_other_aux_dot | 2.5869e-07 | 1.966e-07 | -6.2092e-08 (native_units) |
| p_total_dot | -9.4076e-08 | 4.6377e-06 | 4.7318e-06 (native_units) |
| perception_aux_loc_cosine | 0.017869 | 0.013159 | -0.0047103 (absolute) |
| perception_raw_norm | 0.015533 | 0.017512 | 0.0019791 (native_units) |
| perception_weighted_norm | 0.00077666 | 0.00087561 | 9.8955e-05 (native_units) |
| u_base_dot | 0.00038053 | 0.00032619 | -5.4338e-05 (native_units) |
| u_cosine_to_total | 0.13322 | 0.11269 | -0.020528 (absolute) |
| u_cross_aux_cancellation_pct | -0.067981 | -0.06027 | 0.007711 (percentage_points) |
| u_other_aux_dot | 2.5869e-07 | 1.966e-07 | -6.2092e-08 (native_units) |
| u_total_dot | 0.00038079 | 0.00032639 | -5.44e-05 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 0 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `f3cab27617e9ffb8e06122445159f20775d0342b2628abdb439ac29047b18f4a` / `4bf12548e3a6356a305e8a8a3b8ea58696a37e807d3aac76c68a51b68f27c989`.

Matched source IDs: True (128 rows; SHA256 `bfcb5e2a0a1cdd07100c38f8da1034ab2dc53918977f5550f2fd95e2dba5b02b`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.027893 | 0.024459 | -0.0034337 (native_units) |
| aux_loc_weighted_norm | 0.027893 | 0.024459 | -0.0034337 (native_units) |
| aux_over_main_pct | 33.362 | 29.473 | -3.8898 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 1.6208 | -0.82358 | -2.4444 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 1.2887 | 0.87213 | -0.41653 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.0014799 | -0.0017778 | -0.00029787 (absolute) |
| gen_main_only_descent_first_order | -0.0059227 | -0.0061223 | -0.00019965 (native_units) |
| gen_other_main_cancellation_pct | 0.062138 | 0.063967 | 0.0018288 (percentage_points) |
| gen_raw_norm | 0.076983 | 0.07827 | 0.0012875 (native_units) |
| gen_total_descent_first_order | -0.0058267 | -0.0061728 | -0.00034606 (native_units) |
| gen_weighted_norm | 0.076983 | 0.07827 | 0.0012875 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.12298 | 1.7807 | 1.6577 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 10.268 | 5.6526 | -4.6151 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.00013014 | -9.8654e-05 | 3.1488e-05 (native_units) |
| loc_other_main_cancellation_pct | 0.35246 | 0.49408 | 0.14162 (percentage_points) |
| loc_raw_norm | 0.0040404 | 0.0035204 | -0.00052009 (native_units) |
| loc_total_descent_first_order | -0.00012998 | -9.6897e-05 | 3.3084e-05 (native_units) |
| loc_weighted_norm | 0.032324 | 0.028163 | -0.0041607 (native_units) |
| p_base_dot | 1.4639e-05 | 7.1636e-06 | -7.4754e-06 (native_units) |
| p_cosine_to_total | 0.21566 | 0.14592 | -0.069741 (absolute) |
| p_cross_aux_cancellation_pct | 11.6 | -11.879 | -23.48 (percentage_points) |
| p_other_aux_dot | -1.6982e-06 | 8.51e-07 | 2.5492e-06 (native_units) |
| p_total_dot | 1.2941e-05 | 8.0146e-06 | -4.9262e-06 (native_units) |
| perception_aux_loc_cosine | -0.088126 | 0.055168 | 0.14329 (absolute) |
| perception_raw_norm | 0.013817 | 0.012613 | -0.0012035 (native_units) |
| perception_weighted_norm | 0.00069084 | 0.00063067 | -6.0174e-05 (native_units) |
| u_base_dot | 0.00066658 | 0.00062786 | -3.8722e-05 (native_units) |
| u_cosine_to_total | 0.27443 | 0.29515 | 0.020713 (absolute) |
| u_cross_aux_cancellation_pct | 0.25476 | -0.13554 | -0.3903 (percentage_points) |
| u_other_aux_dot | -1.6982e-06 | 8.51e-07 | 2.5492e-06 (native_units) |
| u_total_dot | 0.00066488 | 0.00062871 | -3.6173e-05 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 0 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `f3cab27617e9ffb8e06122445159f20775d0342b2628abdb439ac29047b18f4a` / `4bf12548e3a6356a305e8a8a3b8ea58696a37e807d3aac76c68a51b68f27c989`.

Matched source IDs: True (128 rows; SHA256 `bfcb5e2a0a1cdd07100c38f8da1034ab2dc53918977f5550f2fd95e2dba5b02b`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.013198 | 0.014185 | 0.0009874 (native_units) |
| aux_loc_weighted_norm | 0.013198 | 0.014185 | 0.0009874 (native_units) |
| aux_over_main_pct | 6.5572 | 5.788 | -0.76916 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.031773 | 0.20395 | 0.23572 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 18.344 | 13.661 | -4.6826 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.004879 | -0.0041198 | 0.00075914 (absolute) |
| gen_main_only_descent_first_order | -0.0083695 | -0.0068628 | 0.0015067 (native_units) |
| gen_other_main_cancellation_pct | 0.95759 | 1.146 | 0.18844 (percentage_points) |
| gen_raw_norm | 0.091926 | 0.083321 | -0.0086053 (native_units) |
| gen_total_descent_first_order | -0.0083721 | -0.0068488 | 0.0015233 (native_units) |
| gen_weighted_norm | 0.091926 | 0.083321 | -0.0086053 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.13732 | 0.06502 | -0.072301 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 3.3304 | 1.5873 | -1.7431 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0040589 | -0.0067051 | -0.0026462 (native_units) |
| loc_other_main_cancellation_pct | 0.24859 | 0.1481 | -0.10048 (percentage_points) |
| loc_raw_norm | 0.022553 | 0.028972 | 0.0064193 (native_units) |
| loc_total_descent_first_order | -0.0040533 | -0.0067008 | -0.0026474 (native_units) |
| loc_weighted_norm | 0.18042 | 0.23178 | 0.051355 (native_units) |
| p_base_dot | 7.1692e-06 | 6.1531e-06 | -1.0161e-06 (native_units) |
| p_cosine_to_total | 0.054361 | 0.042706 | -0.011655 (absolute) |
| p_cross_aux_cancellation_pct | -6.5547 | -8.7199 | -2.1652 (percentage_points) |
| p_other_aux_dot | 4.6992e-07 | 5.3654e-07 | 6.6623e-08 (native_units) |
| p_total_dot | 7.6391e-06 | 6.6896e-06 | -9.495e-07 (native_units) |
| perception_aux_loc_cosine | 0.051263 | 0.059446 | 0.0081835 (absolute) |
| perception_raw_norm | 0.013892 | 0.012726 | -0.0011661 (native_units) |
| perception_weighted_norm | 0.00069458 | 0.00063628 | -5.8304e-05 (native_units) |
| u_base_dot | 0.00012556 | 0.0001466 | 2.1033e-05 (native_units) |
| u_cosine_to_total | 0.047201 | 0.042131 | -0.0050693 (absolute) |
| u_cross_aux_cancellation_pct | -0.37425 | -0.366 | 0.0082504 (percentage_points) |
| u_other_aux_dot | 4.6992e-07 | 5.3654e-07 | 6.6623e-08 (native_units) |
| u_total_dot | 0.00012603 | 0.00014713 | 2.11e-05 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 128 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `47e972babc42b58abb219c910d9f4d27639058341b3f51e54461d55f088aed88` / `e8be14edd8829529fe12287aa317d62b37671b9d263f9ab713c632847ec38b98`.

Matched source IDs: True (128 rows; SHA256 `68d53c3eca3ef21190d12f097b68dfe796d75792c99b316a03ff17780df9a549`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.02968 | 0.041395 | 0.011715 (native_units) |
| aux_loc_weighted_norm | 0.02968 | 0.041395 | 0.011715 (native_units) |
| aux_over_main_pct | 31.13 | 38.024 | 6.8934 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | -0.42041 | -1.981 | -1.5606 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 2.491 | 3.5262 | 1.0352 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.0054375 | 0.0052658 | -0.00017175 (absolute) |
| gen_main_only_descent_first_order | -0.0066937 | -0.0079418 | -0.0012481 (native_units) |
| gen_other_main_cancellation_pct | -0.3271 | -0.36961 | -0.042513 (percentage_points) |
| gen_raw_norm | 0.081682 | 0.088953 | 0.0072708 (native_units) |
| gen_total_descent_first_order | -0.0067219 | -0.0080991 | -0.0013773 (native_units) |
| gen_weighted_norm | 0.081682 | 0.088953 | 0.0072708 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.016253 | -0.37181 | -0.38807 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 3.1114 | 3.7687 | 0.65734 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.00030452 | -0.00049095 | -0.00018643 (native_units) |
| loc_other_main_cancellation_pct | -0.90391 | -0.75021 | 0.15371 (percentage_points) |
| loc_raw_norm | 0.006142 | 0.0078046 | 0.0016626 (native_units) |
| loc_total_descent_first_order | -0.00030447 | -0.00049278 | -0.0001883 (native_units) |
| loc_weighted_norm | 0.049136 | 0.062437 | 0.013301 (native_units) |
| p_base_dot | 1.8816e-06 | -3.3038e-07 | -2.212e-06 (native_units) |
| p_cosine_to_total | 0.057196 | 0.0093893 | -0.047807 (absolute) |
| p_cross_aux_cancellation_pct | -92.524 | — | — (percentage_points) |
| p_other_aux_dot | 1.7409e-06 | 1.0688e-06 | -6.7216e-07 (native_units) |
| p_total_dot | 3.6225e-06 | 7.3838e-07 | -2.8841e-06 (native_units) |
| perception_aux_loc_cosine | 0.092937 | 0.038748 | -0.054189 (absolute) |
| perception_raw_norm | 0.012623 | 0.013326 | 0.00070368 (native_units) |
| perception_weighted_norm | 0.00063114 | 0.00066632 | 3.5184e-05 (native_units) |
| u_base_dot | 0.00090717 | 0.0018862 | 0.00097905 (native_units) |
| u_cosine_to_total | 0.30517 | 0.3863 | 0.081138 (absolute) |
| u_cross_aux_cancellation_pct | -0.19191 | -0.056661 | 0.13525 (percentage_points) |
| u_other_aux_dot | 1.7409e-06 | 1.0688e-06 | -6.7216e-07 (native_units) |
| u_total_dot | 0.00090891 | 0.0018873 | 0.00097838 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 128 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `47e972babc42b58abb219c910d9f4d27639058341b3f51e54461d55f088aed88` / `e8be14edd8829529fe12287aa317d62b37671b9d263f9ab713c632847ec38b98`.

Matched source IDs: True (128 rows; SHA256 `68d53c3eca3ef21190d12f097b68dfe796d75792c99b316a03ff17780df9a549`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.028415 | 0.032917 | 0.0045019 (native_units) |
| aux_loc_weighted_norm | 0.028415 | 0.032917 | 0.0045019 (native_units) |
| aux_over_main_pct | 13.338 | 17.184 | 3.846 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 0.14126 | 0.080908 | -0.060347 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 1.961 | 2.3183 | 0.35734 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.0017393 | 0.0012838 | 0.0030231 (absolute) |
| gen_main_only_descent_first_order | -0.015785 | -0.015471 | 0.00031465 (native_units) |
| gen_other_main_cancellation_pct | 0.23829 | -0.15049 | -0.38878 (percentage_points) |
| gen_raw_norm | 0.12579 | 0.12429 | -0.0015019 (native_units) |
| gen_total_descent_first_order | -0.015763 | -0.015458 | 0.00030487 (native_units) |
| gen_weighted_norm | 0.12579 | 0.12429 | -0.0015019 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.34614 | 0.32823 | -0.017909 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 1.6337 | 1.6161 | -0.017625 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0037077 | -0.0026565 | 0.0010512 (native_units) |
| loc_other_main_cancellation_pct | 0.12696 | -0.10951 | -0.23646 (percentage_points) |
| loc_raw_norm | 0.021542 | 0.018213 | -0.0033292 (native_units) |
| loc_total_descent_first_order | -0.0036948 | -0.0026478 | 0.0010471 (native_units) |
| loc_weighted_norm | 0.17233 | 0.1457 | -0.026633 (native_units) |
| p_base_dot | 7.602e-06 | 1.2612e-05 | 5.0104e-06 (native_units) |
| p_cosine_to_total | 0.061691 | 0.10686 | 0.045168 (absolute) |
| p_cross_aux_cancellation_pct | -5.0683 | -2.0187 | 3.0497 (percentage_points) |
| p_other_aux_dot | 3.853e-07 | 2.546e-07 | -1.3069e-07 (native_units) |
| p_total_dot | 7.9873e-06 | 1.2867e-05 | 4.8797e-06 (native_units) |
| perception_aux_loc_cosine | 0.022463 | 0.012463 | -0.010001 (absolute) |
| perception_raw_norm | 0.012073 | 0.012412 | 0.00033972 (native_units) |
| perception_weighted_norm | 0.00060364 | 0.00062062 | 1.6986e-05 (native_units) |
| u_base_dot | 0.00067518 | 0.000989 | 0.00031381 (native_units) |
| u_cosine_to_total | 0.11085 | 0.1549 | 0.044054 (absolute) |
| u_cross_aux_cancellation_pct | -0.057065 | -0.025743 | 0.031322 (percentage_points) |
| u_other_aux_dot | 3.853e-07 | 2.546e-07 | -1.3069e-07 (native_units) |
| u_total_dot | 0.00067557 | 0.00098925 | 0.00031368 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 256 · seed 17

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `cb674eabd87b49104c62ab8961bc76c28f224a0b9adad380a94dbc089cb6a4a9` / `a9d8e1071c2d36adc7c36f1fca40d438aa494b9e6f3e748693dc2665f27167ce`.

Matched source IDs: True (128 rows; SHA256 `b54576ae2b2e1a5c8ef6875897e49243762d8df3bd26e462ed175994d98a67eb`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.015087 | 0.014064 | -0.0010236 (native_units) |
| aux_loc_weighted_norm | 0.015087 | 0.014064 | -0.0010236 (native_units) |
| aux_over_main_pct | 6.9481 | 7.8304 | 0.88226 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 0.22081 | -0.054263 | -0.27507 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 6.7129 | 4.6126 | -2.1003 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | -0.0016708 | -6.1549e-05 | 0.0016093 (absolute) |
| gen_main_only_descent_first_order | -0.0066911 | -0.0067087 | -1.7579e-05 (native_units) |
| gen_other_main_cancellation_pct | 0.41028 | 0.012022 | -0.39826 (percentage_points) |
| gen_raw_norm | 0.081968 | 0.081912 | -5.6015e-05 (native_units) |
| gen_total_descent_first_order | -0.0066764 | -0.0067124 | -3.5994e-05 (native_units) |
| gen_weighted_norm | 0.081968 | 0.081912 | -5.6015e-05 (native_units) |
| loc_aux_effect_vs_main_descent_pct | -0.10779 | -0.11597 | -0.0081844 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 1.1989 | 1.6839 | 0.48504 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0050604 | -0.0031996 | 0.0018608 (native_units) |
| loc_other_main_cancellation_pct | 0.068044 | 0.0031511 | -0.064893 (percentage_points) |
| loc_raw_norm | 0.025159 | 0.019999 | -0.0051601 (native_units) |
| loc_total_descent_first_order | -0.0050659 | -0.0032033 | 0.0018626 (native_units) |
| loc_weighted_norm | 0.20127 | 0.15999 | -0.041281 (native_units) |
| p_base_dot | 4.7832e-06 | 4.2155e-06 | -5.6762e-07 (native_units) |
| p_cosine_to_total | 0.032169 | 0.036895 | 0.0047255 (absolute) |
| p_cross_aux_cancellation_pct | 3.3138 | 1.1474 | -2.1664 (percentage_points) |
| p_other_aux_dot | -1.5851e-07 | -4.8371e-08 | 1.1013e-07 (native_units) |
| p_total_dot | 4.6247e-06 | 4.1672e-06 | -4.5748e-07 (native_units) |
| perception_aux_loc_cosine | -0.015921 | -0.0054957 | 0.010425 (absolute) |
| perception_raw_norm | 0.013198 | 0.012517 | -0.000681 (native_units) |
| perception_weighted_norm | 0.00065989 | 0.00062584 | -3.405e-05 (native_units) |
| u_base_dot | 0.00025214 | 0.00022729 | -2.485e-05 (native_units) |
| u_cosine_to_total | 0.076664 | 0.089532 | 0.012867 (absolute) |
| u_cross_aux_cancellation_pct | 0.062864 | 0.021282 | -0.041582 (percentage_points) |
| u_other_aux_dot | -1.5851e-07 | -4.8371e-08 | 1.1013e-07 (native_units) |
| u_total_dot | 0.00025198 | 0.00022724 | -2.474e-05 (native_units) |

### exp32_3maps_B2_S3_J8 · offset 256 · seed 29

Individual report validation: passed; pair identity and settings checks: source_seed_equal=True, source_offset_equal=True, ordered_source_ids_equal=True, noise_seeds_equal=True, microbatch_size_equal=True, accumulations_equal=True, report_coordinate_equal=True, result_coordinate_equal=True, metadata_loss_weights_equal=True, measured_weights_equal=True, report_measurement_mode_equal=True, result_measurement_mode_equal=True, tasks_equal=True.

Report SHA256 (1200 / 1800): `cb674eabd87b49104c62ab8961bc76c28f224a0b9adad380a94dbc089cb6a4a9` / `a9d8e1071c2d36adc7c36f1fca40d438aa494b9e6f3e748693dc2665f27167ce`.

Matched source IDs: True (128 rows; SHA256 `b54576ae2b2e1a5c8ef6875897e49243762d8df3bd26e462ed175994d98a67eb`); source indices equal: True.

Coordinate: shared (equal: True; SHA256 `55091982e193f7c35948a92dbc622587e2fadb2e881ec80e5443b149d155e1c7`). Microbatch: 2; accumulations: 64; mode: component_sum; source seed: 20261102; noise seeds: [17, 29].

Metadata weights: `{"alpha_gen_aux_loss": 0.0, "alpha_loc_aux_loss": 1.0, "alpha_loc_loss": 8.0, "alpha_loc_perception_loss": 0.05}`. Measured weights equal: True; values: `{"aux_loc": [1.0], "gen": [1.0], "gen_aux": [0.0], "loc": [8.0], "loc_repa": [0.0], "perception": [0.05000000074505806], "repa": [0.0]}`.

RNG state after measurement: **equal**; differing components: [].

Diagnostic offload metadata (1200 / 1800): `{"metadata.saved_tensor_offload": {"checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence", "enabled": true, "entrypoint": "scripts/probe_multitask_gradients_offload.py", "entrypoint_sha256": "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14", "mechanism": "torch.autograd.graph.saved_tensors_hooks", "original_probe_sha256": "8bf35cab9b54bc76119efd94bf280920d00918fc10eac649e5a377daef42ef2a", "pin_memory": true, "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors", "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards", "torch_version": "2.11.0.dev20260124+cu128"}}` / `{}`. Empty means not reported by that report.

| Metric | Step 1200 | Step 1800 | Δ |
|---|---:|---:|---:|
| aux_loc_raw_norm | 0.020348 | 0.020643 | 0.00029539 (native_units) |
| aux_loc_weighted_norm | 0.020348 | 0.020643 | 0.00029539 (native_units) |
| aux_over_main_pct | 9.2649 | 10.445 | 1.1805 (percentage_points) |
| gen_aux_effect_vs_main_descent_pct | 0.29002 | -0.52038 | -0.8104 (percentage_points) |
| gen_global_main_only_ascent | 0 | 0 | 0 (count) |
| gen_global_total_ascent | 0 | 0 | 0 (count) |
| gen_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| gen_layer_max_cross_main_cancellation_pct | 6.1982 | 5.2414 | -0.95682 (percentage_points) |
| gen_layer_total_ascent_count | 0 | 0 | 0 (count) |
| gen_loc_main_cosine | 0.0025193 | 0.0005476 | -0.0019717 (absolute) |
| gen_main_only_descent_first_order | -0.0067565 | -0.0073007 | -0.00054413 (native_units) |
| gen_other_main_cancellation_pct | -0.62653 | -0.11441 | 0.51212 (percentage_points) |
| gen_raw_norm | 0.081942 | 0.085395 | 0.0034533 (native_units) |
| gen_total_descent_first_order | -0.0067369 | -0.0073386 | -0.00060172 (native_units) |
| gen_weighted_norm | 0.081942 | 0.085395 | 0.0034533 (native_units) |
| loc_aux_effect_vs_main_descent_pct | 0.088463 | 0.030259 | -0.058204 (percentage_points) |
| loc_global_main_only_ascent | 0 | 0 | 0 (count) |
| loc_global_total_ascent | 0 | 0 | 0 (count) |
| loc_layer_main_only_ascent_count | 0 | 0 | 0 (count) |
| loc_layer_max_cross_main_cancellation_pct | 1.0654 | 1.3825 | 0.31703 (percentage_points) |
| loc_layer_total_ascent_count | 0 | 0 | 0 (count) |
| loc_main_only_descent_first_order | -0.0051962 | -0.0039798 | 0.0012164 (native_units) |
| loc_other_main_cancellation_pct | -0.1013 | -0.026211 | 0.075091 (percentage_points) |
| loc_raw_norm | 0.025473 | 0.022301 | -0.0031716 (native_units) |
| loc_total_descent_first_order | -0.0051916 | -0.0039786 | 0.001213 (native_units) |
| loc_weighted_norm | 0.20378 | 0.17841 | -0.025373 (native_units) |
| p_base_dot | 2.892e-06 | 5.83e-06 | 2.938e-06 (native_units) |
| p_cosine_to_total | 0.021524 | 0.0462 | 0.024676 (absolute) |
| p_cross_aux_cancellation_pct | -6.2397 | -3.839 | 2.4008 (percentage_points) |
| p_other_aux_dot | 1.8045e-07 | 2.2381e-07 | 4.3358e-08 (native_units) |
| p_total_dot | 3.0725e-06 | 6.0538e-06 | 2.9814e-06 (native_units) |
| perception_aux_loc_cosine | 0.0137 | 0.01647 | 0.0027697 (absolute) |
| perception_raw_norm | 0.012946 | 0.013166 | 0.0002193 (native_units) |
| perception_weighted_norm | 0.00064732 | 0.00065828 | 1.0965e-05 (native_units) |
| u_base_dot | 0.0003552 | 0.00044911 | 9.3911e-05 (native_units) |
| u_cosine_to_total | 0.0792 | 0.10935 | 0.030149 (absolute) |
| u_cross_aux_cancellation_pct | -0.050804 | -0.049835 | 0.00096912 (percentage_points) |
| u_other_aux_dot | 1.8045e-07 | 2.2381e-07 | 4.3358e-08 (native_units) |
| u_total_dot | 0.00035538 | 0.00044933 | 9.3954e-05 (native_units) |

