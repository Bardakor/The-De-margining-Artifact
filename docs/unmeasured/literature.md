# Literature — The Unmeasured Remainder

Checked on 2026-11-14. 115 entries, all in `paper-unmeasured/refs.bib`.

How each entry was verified:

- **DOI entries**: the Crossref record of the DOI (publisher-deposited metadata,
  queried at `api.crossref.org/works/<doi>`) matched authors, year, title and
  venue. The URL listed is the DOI link. Where the online and print years
  differ, the bib uses the print (volume) year and notes the online year.
- **Entries without a DOI**: the page named in the "Verified at" column (the
  proceedings, author, institutional repository, arXiv or library catalogue
  page).
- **Summaries**: written from abstracts read during the check (Crossref
  abstract, publisher or repository page). Full texts were mostly not read. A
  number is given only where an abstract or the publisher page states it.

Module codes follow design §5: M1 reliability, M2 structure, M3 RAPM,
M4 unmeasured share, M5 multiverse, M6 recognition, M7a on-ball value
visibility, M7b big-game trait, M8 synthesis, M9 recovery study, Tax = taxonomy,
Data = data sources.

---

## 1. Randomness in football results

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `lames2018chance` | Coded Bundesliga and Premier League goals for six "chance" markers (deflections, rebounds, post hits, long shots, keeper or defender involvement). By secondary summaries, close to half of goals involve at least one marker. | Tax, M1, M8 | https://doi.org/10.1007/s12662-018-0518-z |
| `wunderlich2021randomness` | Coded 7,263 Premier League goals (2012/13–2018/19) for the same six markers. About 46% of goals involve chance, falling over the seasons, and chance matters more for weaker teams and at level scores. Authors: Wunderlich, **Seck** & Memmert. | Tax, M1, M8 | https://doi.org/10.1080/02640414.2021.1930685 |
| `aoki2017luck` | A skill-versus-luck coefficient across 1,503 seasons of 198 leagues in four sports. Luck is substantial even in the most competitive leagues, which is why complex predictors barely beat simple ones. | M9, Tax | https://doi.org/10.1145/3097983.3098045 |
| `brechot2020randomness` | Short-run results in European club football are heavily shaped by randomness. The paper proposes xG-based charts to spot when results misstate underlying performance. | M1, Tax | https://doi.org/10.1177/1527002519897962 |
| `heuer2010poisson` | Bundesliga goals behave approximately as independent Poisson events with pre-match rates. Small correlations add draws, and the over-dispersion comes from differences in team strength. Quantifies the limits of match predictability. | M3, M9, Tax | https://doi.org/10.1209/0295-5075/89/38007 |
| `skinner2009experiments` | Treats a match as an experiment for picking the better team. For typical scorelines the chance that the result misleads is large, and no realistic rule change fixes it. | M9, Tax | https://doi.org/10.1080/02664760802715922 |

## 2. Reliability, measurement error, bootstrap, shrinkage

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `franks2016metaanalytics` | "Meta-metrics" for sports statistics: stability (signal over time), discrimination (spread between players) and independence (new information). Demonstrated on NBA and NHL. This is the framework M1/M2 operationalise. | M1, M2 | https://doi.org/10.1515/jqas-2016-0098 |
| `spearman1904proof` | Introduces the correction of a correlation for unreliability of both measures (disattenuation). | M4 | https://doi.org/10.2307/1412159 |
| `spearman1910faulty` | Spearman's statement of the reliability step-up for a lengthened test (with Brown 1910: Spearman–Brown). | M1 | https://doi.org/10.1111/j.2044-8295.1910.tb00206.x |
| `brown1910experimental` | Brown's companion derivation of the same step-up formula. | M1 | https://doi.org/10.1111/j.2044-8295.1910.tb00207.x |
| `lames2007reliable` | Argues that performance indicators in game sports are inherently unstable because performance emerges from interaction with the opponent. Recommends modelling and simulation over a search for stable indicators. | M1, M7b, Tax | https://doi.org/10.1080/24748668.2007.11868388 |
| `charles2005attenuation` | Disattenuated correlations can exceed 1 and have awkward sampling behaviour. Gives confidence sets for the corrected value. Directly relevant to M4's R²_true. | M4 | https://doi.org/10.1037/1082-989X.10.2.206 |
| `shrout1979icc` | The standard forms of the intraclass correlation and when to use each. | M2 (team ICC), M1 | https://doi.org/10.1037/0033-2909.86.2.420 |
| `efron1979bootstrap` | The bootstrap. | M1, M3, M8 | https://doi.org/10.1214/aos/1176344552 |
| `efron1975stein` | Stein / empirical-Bayes shrinkage of many means toward a common value, illustrated on batting averages. | M5 (shrinkage dimension) | https://doi.org/10.1080/01621459.1975.10479864 |
| `gregson2010variability` | Tracking data on 485 Premier League outfield players. Match-to-match CV is about 16% for high-speed running and about 31% for sprint distance, higher for central players. Physical outputs are noisy per match. Publisher metadata lists the 4th author as "V. Salvo"; the author is Valter Di Salvo, and refs.bib uses that name. | Tax, M1 | https://doi.org/10.1055/s-0030-1247546 |

## 3. Plus-minus and RAPM

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `hvattum2019review` | Review of advanced plus-minus ratings across team sports. The literature is fragmented, with many open questions on individual evaluation. | M3 | https://doi.org/10.2478/ijcss-2019-0001 |
| `saebo2015transfer` | First regression-based (plus-minus) player rating for football. The rating explains a substantial share of transfer fees and points to some market inefficiencies. | M3 | https://www.ntnu.no/ojs/index.php/nikt/article/view/5265 |
| `saebo2019financial` | Goal-difference-based player ratings feed a match model and Monte Carlo simulation to value Premier League transfers financially. | M3 | https://doi.org/10.3233/JSA-170235 |
| `kharrat2020plusminus` | Adapts plus-minus to football. Proposes expected-goals (xGPM) and expected-points (xPPM) targets because the goal-difference target is very sparse. | M3 | https://doi.org/10.1016/j.ejor.2019.11.026 |
| `sill2010rapm` | Ridge-regularised adjusted plus-minus (RAPM) with out-of-sample evaluation. Regularisation nearly doubles predictive accuracy over plain APM. | M3 | https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing |
| `macdonald2011apm` | Regression-based adjusted plus-minus for NHL players. | M3 | https://doi.org/10.2202/1559-0410.1284 |
| `macdonald2012ridge` | Ridge APM for the NHL. Collinearity of linemates and low scoring make OLS error bars wide, ridge narrows them, and richer events (shots, Fenwick, Corsi) shrink errors further. | M3 | https://doi.org/10.1515/1559-0410.1447 |
| `hoerl1970ridge` | Ridge regression. | M3, M4 | https://doi.org/10.1080/00401706.1970.10488634 |
| `gramacy2013hockey` | Regularised logistic plus-minus for hockey with careful shrinkage. Most players are **not measurably different from zero**, and a few stars stand out. | M3, M4, M9 | https://doi.org/10.1515/jqas-2012-0001 |
| `schultze2018weighted` | A weighted plus/minus for football that adjusts for opponent strength and goal importance, demonstrated on three Bundesliga teams. | M3 | https://doi.org/10.3233/JSA-170225 |
| `hvattum2020offdef` | Splits football plus-minus into offensive and defensive components, including age, red-card and home-advantage effects. This is the template for M3's separate O/D effects. | M3, Tax | https://doi.org/10.3390/app10207345 |
| `gelade2020events` | **Closest prior to M4.** Event-level statistics explain **22% to 38%** of the variance in football plus-minus ratings, depending on position, and add only marginally to the ratings' predictive power. Abstract only; the per-position split was not retrievable. | M4, M3, Tax | https://doi.org/10.3233/JSA-200432 |
| `hvattum2021bottomup` | Compares top-down (plus-minus) and bottom-up (VAEP-style action values) ratings. Plus-minus was more reliable across data sets and gave better match forecasts. | M3, M4, M7a | https://doi.org/10.2478/ijcss-2021-0002 |
| `arntzen2021ratings` | Elo team ratings and plus-minus player ratings forecast matches about equally well, and the two together beat either alone. | M3, Data (Elo) | https://doi.org/10.1177/1471082X20929881 |
| `paige1982lsqr` | LSQR, the sparse least-squares solver M3 uses. | M3 | https://doi.org/10.1145/355984.355989 |

## 4. Valuing events, off-ball play and defending

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `decroos2019vaep` | VAEP: values every on-ball action by its change in short-horizon scoring and conceding probabilities. | M7a, Tax | https://doi.org/10.1145/3292500.3330758 |
| `singh2019xt` | **Grey literature (blog).** Expected threat: a Markov grid that values ball moves by the change in the chance of scoring within the next few actions. The page is undated; 2019 is the year peer-reviewed work cites (Fernández et al. 2021 reference list). | M7a (xt_added), Data | https://karun.in/blog/expected-threat.html |
| `fernandez2019epv` | Expected possession value from tracking data, decomposed into pass, drive and shot components (conference version). | M7a, Tax | https://www.sloansportsconference.com/research-papers/decomposing-the-immeasurable-sport-a-deep-learning-expected-possession-value-framework-for-soccer |
| `fernandez2021epv` | Journal version of EPV. Possession value is decomposed into pass, ball-drive and shot sub-models with deep-learning surfaces. | M7a, Tax | https://doi.org/10.1007/s10994-021-05989-6 |
| `spearman2018beyond` | Off-ball scoring opportunity (OBSO): a physics-based pitch-control model on tracking data that credits players for being in dangerous positions they never receive the ball in. | Tax | https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals (author-uploaded PDF; also cited in arXiv:2011.09426) |
| `fernandez2018wide` | Measures space occupation and space generation (dragging defenders) from tracking data, with examples from one La Liga match. The proceedings PDF is anonymised. The Sloan page lists Fernández, and the co-author Bornn is given in the authors' own arXiv:2011.09426 reference list. | Tax | https://sloansportsconference.com/research-papers/wide-open-spaces-a-statistical-technique-for-measuring-space-creation-in-professional-soccer |
| `power2017passes` | Values passes by risk and reward from tracking data, so pass choice can be judged relative to the options available. | Tax | https://doi.org/10.1145/3097983.3098051 |
| `vanroy2020xtvaep` | **Blog post** (the AAAI-20 workshop version was not verified; cited as the DTAI blog post). xT and VAEP rank players differently. xT leans to playmaking and VAEP to shooting, both differ from goals/assists, and xT cannot value defensive actions. | M7a, M5 | https://dtai.cs.kuleuven.be/sports/blog/valuing-on-the-ball-actions-in-soccer-a-critical-comparison-of-xt-and-vaep |
| `merhej2021defensive` | DAxT values tackles and interceptions by the threat they stop. It starts from the observation that event metrics mostly value attacking play. | Tax, M7a | https://doi.org/10.1145/3447548.3467090 |
| `bauer2021counterpressing` | Detects counterpressing from synchronised positional and event data, then quantifies its success drivers over six Bundesliga seasons. | Tax | https://doi.org/10.1007/s10618-021-00763-7 |
| `pappalardo2019playerank` | PlayeRank: a role-aware, multi-dimensional player rating from event logs, validated against professional scouts' judgements. | M2, M6, Tax | https://doi.org/10.1145/3343172 |
| `aalbers2019roles` | Classifies player roles from event data, beyond the nominal lineup position. | Tax, M2 | https://doi.org/10.1007/978-3-030-17274-9_3 |
| `bialkowski2014roles` | Infers formations and roles from tracking data at scale. | Tax | https://doi.org/10.1109/ICDM.2014.133 |

## 5. Expected goals and finishing

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `pollard1997strategies` | Early shot-quality model: scoring probability from shot location and circumstances, used to rate playing strategies. | H8, Data (zone-xG) | https://doi.org/10.1111/1467-9884.00108 |
| `rathke2017xg` | Shot-location xG on one Bundesliga and one Premier League season. xG tracks mid-table teams well, while top teams over-achieve and bottom teams under-achieve. | H8, M1 | https://doi.org/10.14198/jhse.2017.12.Proc2.05 |
| `robberechts2020xg` | How much and which data xG needs: location is among the strongest features, complex models need about five seasons, old data loses little, and mixing data providers should be avoided. Relevant to zone-xG versus StatsBomb xG (H8). | H8, Data | https://doi.org/10.1007/978-3-030-64912-8_2 |
| `anzer2021xg` | xG from synchronised positional and event data on 105,627 Bundesliga shots, more accurate than prior published models. Sets the tracking-data ceiling on what coarse zones lose. | H8, Tax | https://doi.org/10.3389/fspor.2021.624475 |
| `mead2023xg` | Adds player/team-ability and psychological features to xG, which add information. xG predicts future team success better than traditional statistics. The closest thing to a review we could verify (see "Not found"). | H8, M1 | https://doi.org/10.1371/journal.pone.0282295 |
| `scholtes2024bayesxg` | Bayesian hierarchical xG on public StatsBomb shots. Position effects fade once richer features enter, but player-level finishing effects persist. | M1, M4, Tax | https://doi.org/10.3389/fspor.2024.1348983 |
| `davis2024finishing` | **Preprint.** "Goals minus xG" is a poor finishing measure because of outcome variance, small samples and xG model bias. Consistently beating xG needs both volume and skill, and standard xG understates elite finishers. | M1, M4, Tax | https://arxiv.org/abs/2401.09940 |

## 6. Uncertainty in rankings

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `goldstein1996league` | League tables of institutions carry large model-based uncertainty. Ranks should be reported with intervals, not as point orderings. | M8, H1 | https://doi.org/10.2307/2983325 |
| `hall2009bootstrap` | **Design-relevant.** The standard n-out-of-n bootstrap of an empirical ranking can be inconsistent, especially when true values are close or tied. An m-out-of-n bootstrap restores consistency, with m tuned empirically. | M8, H1, H2 | https://doi.org/10.1214/09-AOS699 |
| `mogstad2024ranks` | Marginal and simultaneous confidence sets for ranks under weak assumptions, computable for many units. | M8 | https://doi.org/10.1093/restud/rdad006 |
| `kendall1939rankings` | Kendall's coefficient of concordance W for m rankings. | M8 | https://doi.org/10.1214/aoms/1177732186 |

## 7. Multiverse, many analysts, sensitivity analysis, composite indicators

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `steegen2016multiverse` | Multiverse analysis: run every reasonable data-processing path and show how conclusions move. | M5 | https://doi.org/10.1177/1745691616658637 |
| `simonsohn2020specification` | Specification-curve analysis: estimate all reasonable specifications, plot them, and test jointly. | M5 | https://doi.org/10.1038/s41562-020-0912-z |
| `silberzahn2018many` | **Football data.** 29 teams (61 analysts) asked whether referees give more red cards to dark-skin-toned players: odds ratios ranged from 0.89 to 2.93, 20 teams found a significant effect and 9 did not. Neither expertise nor priors explained the spread. | M5, H4 | https://doi.org/10.1177/2515245917747646 |
| `breznau2022many` | 73 teams testing one social-science hypothesis on the same data diverged widely. Over 95% of the variance in their numerical results was unexplained by coded analytic decisions. | M5, H4 | https://doi.org/10.1073/pnas.2203150119 |
| `sobol2001sensitivity` | Variance-based (Sobol) sensitivity indices and Monte Carlo estimators. | M5 | https://doi.org/10.1016/S0378-4754(00)00270-6 |
| `saltelli2002sensitivity` | Efficient designs for computing first-order and total sensitivity indices. | M5 | https://doi.org/10.1016/S0010-4655(02)00280-1 |
| `saltelli2010total` | Recommended estimators for the total-order index. | M5, H5 | https://doi.org/10.1016/j.cpc.2009.09.018 |
| `saltelli2008primer` | Textbook on global sensitivity analysis (Wiley; online December 2007, print 2008). | M5 | https://doi.org/10.1002/9780470725184 |
| `homma1996importance` | Introduces the total-effect sensitivity index. | M5, H5 | https://doi.org/10.1016/0951-8320(96)00002-6 |
| `saisana2005composite` | Uncertainty and sensitivity analysis applied to a composite indicator to test how robust its country ranking is. This is the closest methodological template for M5 applied to a "best player" index. | M5, M2 | https://doi.org/10.1111/j.1467-985X.2005.00350.x |
| `paruolo2013voodoo` | **Design-relevant.** In a composite indicator, a variable's nominal weight rarely matches its actual importance (main effect). Correlation among components can make the stated importance unreachable. This is the formal version of the V1 double-counting critique and of M2's index. | M2, M5 | https://doi.org/10.1111/j.1467-985X.2012.01059.x |
| `oecd2008composite` | OECD/JRC handbook on building composite indicators (normalisation, weighting, aggregation, robustness). | M2, M5 | https://doi.org/10.1787/9789264043466-en |

## 8. Structure: factors, dependence, effective dimension

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `horn1965parallel` | Parallel analysis: retain factors whose eigenvalues exceed those of random data. | M2 | https://doi.org/10.1007/BF02289447 |
| `szekely2007dcor` | Distance correlation, which is zero only under independence and so detects non-linear dependence. | M2 | https://doi.org/10.1214/009053607000000505 |
| `joreskog1967ml` | Maximum-likelihood factor analysis. | M2 | https://doi.org/10.1007/BF02289658 |
| `gao2017dimensionality` | **Preprint.** Uses the participation ratio of covariance eigenvalues as a continuous dimensionality measure (1 = one dimension, M = variance spread evenly). | M2 | https://doi.org/10.1101/214262 |
| `roy2007effective` | Effective rank: the exponential of the spectral entropy of the singular values. An alternative effective-dimension measure to report alongside the participation ratio. | M2 | https://infoscience.epfl.ch/record/110188 |

## 9. Recognition, awards, market value, ranking models, relative importance

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `coupe2018ballondor` | FIFA Ballon d'Or jury votes. Voters favour own country, national team, continent and league ("similarity" bias), but the effect on totals is small and decisive only when top candidates are close. Voters who back one favourite are *more* likely to give points to his rival. | M6, H6 | https://doi.org/10.1111/obes.12201 |
| `johnson2022voting` | Best FIFA Men's Player votes (captains, coaches, media). Cultural similarity raises voting bias, and media voters are least biased. The body (per publisher page) reports that performance and player effects explain about 75% of vote variation, much of it unobserved player-specific effects. | M6, H6 | https://doi.org/10.1371/journal.pone.0270546 |
| `anderson2020celebrity` | Ballon d'Or history plus a fan survey: certain kinds of players are much more likely to win, votes are concentrating over time, and fan support behaves like partisanship. | M6, H6 | https://doi.org/10.1017/S1537592719002391 |
| `olmeta2026ballondor` | **Undergraduate journal.** Ridge regression of 2025 Ballon d'Or vote shares on statistics and trophies. Adding trophy indicators improves ranking fidelity. Weak evidence, but the only direct study of the current (journalist-voted) award that we found. | M6, H6 | https://journals.library.wustl.edu/wujur/article/id/9230/ |
| `herm2014crowd` | Transfermarkt community valuations predict actual transfer fees well. Studies which talent-related and external-judgement attributes drive them. | M6, Data | https://doi.org/10.1016/j.smr.2013.12.006 |
| `muller2017market` | Data-driven market-value estimates (4,217 players, top five leagues) match crowd valuations in accuracy while being transparent and updatable. | M6 | https://doi.org/10.1016/j.ejor.2017.05.005 |
| `peeters2018wisdom` | Transfermarkt squad valuations forecast international results better than FIFA ranking or Elo. | M6, Data | https://doi.org/10.1016/j.ijforecast.2017.08.002 |
| `franck2012superstar` | In German football, both talent (measured by contribution to winning) and popularity raise players' market value, beyond each other. This is Rosen-versus-Adler "superstar" evidence. | M6, H6 | https://doi.org/10.1111/j.1465-7295.2010.00360.x |
| `plackett1975permutations` | The Plackett model for ranked data (Plackett–Luce). | M6 | https://doi.org/10.2307/2346567 |
| `luce1959choice` | Luce's choice axiom, the basis of the Plackett–Luce model. | M6 | https://openlibrary.org/works/OL4319279W |
| `hunter2004mm` | MM algorithms for fitting Bradley–Terry and Plackett–Luce models. | M6 | https://doi.org/10.1214/aos/1079120141 |
| `bradley1952paired` | The Bradley–Terry paired-comparison model. | M6, M8 | https://doi.org/10.2307/2334029 |
| `budescu1993dominance` | Dominance analysis for relative importance of predictors. | M6 | https://doi.org/10.1037/0033-2909.114.3.542 |
| `lindeman1980multivariate` | Textbook origin of the LMG (averaging-over-orderings) R² decomposition. | M6 | https://ci.nii.ac.jp/ncid/BA1075484X |
| `gromping2007relimp` | Compares variance-decomposition importance measures and recommends LMG/Shapley-type averaging. | M6 | https://doi.org/10.1198/000313007X188252 |
| `gromping2006relaimpo` | Software and exposition of LMG and related relative-importance metrics. | M6 | https://doi.org/10.18637/jss.v017.i01 |
| `shapley1953value` | The Shapley value. | M6 | https://doi.org/10.1515/9781400881970-018 |

## 10. Big games, clutch, hot hand, pressure

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `gilovich1985hothand` | The classic finding that perceived basketball shooting streaks are consistent with chance. | M7b, H7 | https://doi.org/10.1016/0010-0285(85)90010-6 |
| `miller2018hothand` | Conditioning on streaks in finite sequences biases the naive test toward "no hot hand". Correcting it reverses the 1985 conclusion. A warning for any streak or clutch test on short sequences. | M7b, H7 | https://doi.org/10.3982/ECTA14943 |
| `bransen2019choke` | Football: a learned model of in-match mental pressure (from game context), then rates how players' choices and executions change under it. The only football player-level clutch study found. It reports differences but no test of stability over time. | M7b, H7 | https://dtai.cs.kuleuven.be/sports/ssac19/ |
| `gonzalezdiaz2012tennis` | Professional tennis: players differ measurably in how much better they play on important points. A "clutch" trait is detectable in a sport with many high-stakes points per player. | M7b, H7 | https://doi.org/10.1016/j.jebo.2012.09.021 |
| `jordet2008avoidance` | Penalty shootouts (World Cup, Euros, Champions League; 359 kicks): kicks where a miss loses produce more avoidance behaviour and worse outcomes than kicks where a goal wins. | Tax (psychological) | https://doi.org/10.1123/jsep.30.4.450 |
| `apesteguia2010pressure` | Penalty shootouts as a natural experiment: reported a first-kicker advantage. | Tax | https://doi.org/10.1257/aer.100.5.2548 |
| `kocher2012pressure` | Extends that sample from 129 to 540 shootouts and finds **no** significant first-mover advantage. A failed replication. | Tax, M7b | https://doi.org/10.1287/mnsc.1120.1516 |

## 11. Teams, chemistry, leadership, coaches, context

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `mukherjee2019shared` | Prior shared *success* between teammates predicts wins beyond individual skill in several sports, the Premier League included (secondary summary: +2 to 7 points of prediction accuracy). | Tax, M3 | https://doi.org/10.1038/s41562-018-0460-y |
| `lecoz2025shared` | 19,721 matches: shared experience has **no independent effect** on results once team strength is controlled. At pass level, shared experience with successful prior interactions predicts pass success. | Tax, M3 | https://doi.org/10.1186/s40537-025-01239-x |
| `fransen2014captain` | 4,451 players and coaches in nine team sports: the captain is often not perceived as the main leader, and leadership is spread across informal leaders. | Tax (captaincy proxy) | https://doi.org/10.1080/02640414.2014.891291 |
| `carron2002cohesion` | Meta-analysis of 46 studies: cohesion and performance are moderately to strongly related, with direction not identified. | Tax | https://doi.org/10.1123/jsep.24.2.168 |
| `ingersoll2017diversity` | Champions League 2003–2012: more linguistically diverse squads outperform less diverse ones after controlling for talent. | Tax | https://doi.org/10.3233/JSA-170052 |
| `muehlheusser2018managers` | Bundesliga manager turnover: managers from the top of the ability distribution gain considerably more points than those at the bottom. Managers also shape playing style. | Tax, M3 | https://doi.org/10.1177/1527002516674760 |
| `lago2009possession` | Possession depends on match location, opponent quality and current score. Event counts are context-dependent. | Tax, M2 | https://doi.org/10.1080/02640410903131681 |
| `fischer2021ghost` | German ghost games at the end of 2019/20: home advantage fell in the top division only (not in the second and third), and the fall tracks lost stadium occupancy. Does not cover 2020/21. | M3, Tax | https://doi.org/10.1177/15270025211026552 |
| `bryson2021crowd` | Covid-19 empty stadiums (2019/20 only): referees gave away teams fewer yellow cards, narrowing the home–away card gap; no significant change in home wins, goal difference or total goals. | M3, Tax | https://doi.org/10.1016/j.econlet.2020.109664 |

## 12. Physical output, availability, age

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `hagglund2013injuries` | 24 European clubs over 11 seasons: lower injury burden and higher match availability go with higher league rank, more points per match and better European results. | Tax | https://doi.org/10.1136/bjsports-2013-092215 |
| `bradley2009hir` | Describes high-intensity running in Premier League matches and how it varies with playing position. | Tax | https://doi.org/10.1080/02640410802512775 |
| `carling2013physical` | Argues that physical match outputs (distances, high-speed running) are hard to interpret on their own and must be read in tactical and match context. | Tax | https://doi.org/10.1007/s40279-013-0055-8 |
| `dendir2016peak` | WhoScored ratings (2010/11–2014/15, four top leagues). Per the abstract, forwards peak at about 25 and defenders at about 27. | M5 (time aggregation), Tax | https://doi.org/10.3233/JSA-160021 |

## 13. Elo

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `elo1978rating` | The Elo rating system (Arco, New York, 1978). | Data (team_elo) | https://openlibrary.org/books/OL4734793M |
| `hvattum2010elo` | Uses Elo ratings, including a goal-difference variant, as covariates for predicting football match results. | Data (team_elo), M3 | https://doi.org/10.1016/j.ijforecast.2009.10.002 |

## 14. Data sources and reviews

| Key | What it shows | Modules | Verified at |
|---|---|---|---|
| `transfermarktdatasets` | The Transfermarkt-derived CSVs (games, appearances, lineups, events, valuations). CC0-1.0 per the repo. See data-licences.md for the upstream caveat. | Data | https://github.com/dcaribou/transfermarkt-datasets |
| `statsbombopendata` | StatsBomb open event data. Use is governed by the Public Data User Agreement, with logo attribution required (data-licences.md). | Data, M7a, H8 | https://github.com/statsbomb/open-data |
| `pappalardo2019dataset` | A public, openly licensed set of spatio-temporal match events (Wyscout) from several competitions. Another public event-data tier, not used here. | Tax, Data | https://doi.org/10.1038/s41597-019-0247-7 |
| `sarmento2018review` | Systematic review of football match-analysis research 2012–2016 (performance indicators, contextual variables). | Tax | https://doi.org/10.1007/s40279-017-0836-6 |

---

## Dropped (unverified) and not found

Dropped because they could not be verified against a live source, or replaced
by a verified alternative:

- **A systematic review of xG.** Not found. Searches for a PRISMA-style review
  dedicated to xG returned only broader football reviews (e.g. Sarmento et al.
  2018, included) and single-model papers. `mead2023xg` is cited for model
  comparison instead.
- **Rosenbaum (2004), the original adjusted plus-minus (82games.com).** Not
  checked. `sill2010rapm` and the peer-reviewed APM papers cover the method.
- **Rampinini et al. (2007), "Variation in top level soccer match performance"
  (IJSM).** The DOI resolved (10.1055/s-2007-965158), but no abstract could be
  read to summarise it. Left out rather than summarised from memory.
- **The Premier League passing-network familiarity study** (players pass more
  to teammates they already know, PMC6446935). PMC blocked the fetch and the
  Crossref search did not find it. `lecoz2025shared` and
  `mukherjee2019shared` cover familiarity.
- **Recanatesi et al. (2022), Patterns** (participation ratio). The DOI exists,
  but the definition could not be confirmed (publisher blocked).
  `gao2017dimensionality` is used instead.
- **The Gelade & Hvattum (2020) per-position numbers.** The entry is
  verified, but the full text was refused (publisher and mirror). Only the
  22–38% range from the abstract is used, not which position is high or low.
- **Rosen (1981), Adler (1985), the superstar economics.** Not verified
  (JSTOR DOI lookup failed). They are reached through `franck2012superstar`.
- **Transfermarkt's own terms of use.** transfermarkt.com refuses scripted
  fetches. See data-licences.md.

Spelling corrected against the publisher: Wunderlich's co-author is **Alessandro
Seck**. Kharrat et al. author order is Kharrat, McHale, López Peña.
