# ccc40 run log

```
==============================================================================================================
SECTION 7 -- REPORTABLE SET
==============================================================================================================
scored edges in : 60,404
reportable out  : 22
partners represented: ['B', 'CD4_malignant', 'CD4_reactive', 'CD8', 'F_apCAF', 'F_inflammatory', 'F_reticular', 'Keratinocyte', 'Mac_FOLR2', 'Mac_SPP1_TREM2', 'Mac_infl', 'Mono_moDC']
negative controls leaked into the set: 0 (asserted 0)
shuffle-recurrent edges leaked into the set: 0 (asserted 0); 0 dropped at gate 5
within-donor shuffle overlap (commentary): [2, 4, 2]

--- per-gate drop counts (applied in order) ---
                   gate  n_in  n_out  n_dropped  n_failing_alone
     1_levels_claimable 60404  60404          0                0
   2_donor_reproducible 60404  42673      17731            17731
      3_paired_contrast 42673    626      42047            59533
          4_not_ambient   626    626          0                0
5_not_shuffle_recurrent   626    626          0               25
           6_top_decile   626     22        604            54363

--- reportable set (top 25) ---
        source         target ligand_complex receptor_complex  specificity_rank  donor_frac  n_studies_recovered  contrast_lfc  contrast_fdr
 CD4_malignant            CD8           CD70             CD27         0.0001473      0.8519                    6         1.198     6.565e-05
 CD4_malignant              B           CD70        TNFRSF13B         0.0005791        0.65                    3         1.198     6.565e-05
 CD4_malignant      Mac_FOLR2            SPN          SIGLEC1         0.0006214        0.95                    5        0.5567         5e-06
     Mac_FOLR2  CD4_malignant           C1QB            C1QBP           0.00121           1                    5        0.4471       0.02128
 CD4_malignant   CD4_reactive           CD70             CD27          0.001995      0.8621                    6         1.198     6.565e-05
 CD4_malignant F_inflammatory        TNFSF14         TNFRSF6B          0.002061      0.7778                    2        0.9809     2.799e-06
 CD4_malignant   Keratinocyte         S100A4            ERBB3          0.002342           1                    6        0.4599       0.03163
F_inflammatory  CD4_malignant          ICAM1            IL2RG          0.003061           1                    2        0.2829      0.004094
 CD4_malignant        F_apCAF        TNFSF14         TNFRSF6B          0.003239      0.6842                    4        0.9809     2.799e-06
 CD4_malignant Mac_SPP1_TREM2            SPN          SIGLEC1          0.003637       0.625                    4        0.5567         5e-06
 CD4_malignant   Keratinocyte          HMGB1             SDC1          0.003963      0.5909                    5        0.5093       0.03946
 CD4_malignant   Keratinocyte          PTHLH            ADRB2          0.004768         0.5                    5         2.871      2.45e-13
 CD4_malignant F_inflammatory            MIF            ACKR3          0.007282      0.7778                    2        0.6899      1.08e-09
 CD4_malignant   Keratinocyte         S100A4            ERBB2          0.009525      0.7273                    5        0.4599       0.03163
 CD4_malignant       Mac_infl           CD99            PILRA           0.01025           1                    5        0.4355      0.002484
 CD4_malignant F_inflammatory          GNAI2            EDNRB           0.01062           1                    2        0.4335     0.0006806
     Mono_moDC  CD4_malignant          HBEGF             CD82           0.01068        0.75                    6         0.572      0.001353
 CD4_malignant    F_reticular            MIF            ACKR3           0.01093      0.7619                    5        0.6899      1.08e-09
 CD4_malignant Mac_SPP1_TREM2        TNFSF14         TNFRSF14           0.01214       0.625                    4        0.9809     2.799e-06
      Mac_infl  CD4_malignant           IL10             CD70           0.01225      0.6842                    5         1.198     6.565e-05
 CD4_malignant   Keratinocyte           RTN4             GJB2           0.01266      0.9091                    5        0.4042     4.693e-06
     Mac_FOLR2  CD4_malignant           IL10             CD70           0.01445         0.7                    5         1.198     6.565e-05
```

```
==============================================================================================================
SECTION 7 -- FIGURES
==============================================================================================================
1 ccc40_dotplot.svg : x = partner, y = interaction, colour = -log10 specificity_rank, size = fraction of donors recovering the edge; one panel per direction, top 20.
2 ccc40_heatmap.svg : interactions x partner, shared vmin/vmax, grey = structurally absent, * = paired-contrast FDR < 0.05, top 15.
3 ccc40_landing.svg : RESOLUTION_TEST x level, median within-donor rank, winner annotated with the donor-jackknife win %.
4 ccc40_panel.svg   : forced curated panel, size = DETECTION proportion.
no network / chord / circos, no UMAP, no violin or ridge plot.
```

```
==============================================================================================================
FIGURE 1b -- CURATED EDGES BY BIOLOGICAL PROGRAM
==============================================================================================================
file            : ccc40_dotplot_programs.svg
programs        : 10, 36 curated edges
encoding        : as figure 1 (colour = -log10 specificity_rank, size = donor frac); rows blocked by program, no top-N cut
```
