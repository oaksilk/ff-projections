- FEEDBACK BELOW --

My overall grade is C+ for the tool as it stands today. The underlying modeling foundation earns a B.
It has meaningful predictive signal, a sensible choice of modeling techniques, and a useful interface. However, its historical evaluation contains information leakage, its published forecasts differ from the forecasts evaluated in the backtest, and some explanations and recommendations imply more certainty than the implementation supports.
I would use it as a second opinion today. I would not yet rely on it as my sole start/sit authority or claim that it beats expert consensus.
I reviewed the model, feature engineering, data loaders, backtest, market integration, publishing code, automation, and live interface. I rebuilt the historical features, independently recalculated the saved results, and retrained the 2025 weeks 13–18 prediction block: all 2,096 predictions matched the saved predictions exactly. I also ran a separate chronological check of the uncertainty model. I did not change the product.
The grades below prioritize predictive integrity and decision usefulness. They are judgments against a dependable forecasting product, rather than grades for how much was accomplished in one day.
Area	Grade	Assessment
Core modeling approach	B	Sensible, regularized statistical models using relevant inputs
Historical evaluation	C−	Substantial effort, but important fairness and leakage problems
Ranges and probabilities	B−	Encouraging interval calibration; live and head-to-head probabilities remain unvalidated
Market integration	C	Thoughtful approach, with assumptions and weighting that need testing
Explanations	C	Useful idea; currently misaligned with the forecast users see
Interface and usability	B	Focused, readable, and easy to explore
Operational reliability	C+	Good automation foundation; incomplete freshness and archival safeguards
Overall	C+	Useful prototype with material trustworthiness gaps


What the model actually does
The implementation builds 92 features from player history, team tendencies, opponent results, injuries, quarterback context, weather, and betting lines.
It then trains:
- Eight models predicting individual statistics, including receptions, yards, touchdowns, and fumbles.
- Three models predicting fantasy points directly, one for each scoring format.
- Fifteen quantile models estimating outcome ranges across the three formats.
The point projection averages the component-based forecast and direct forecast. Available player props then influence the published projection.
That is a reasonable architecture. Gradient boosting is a credible choice for these tabular inputs. The conservative tree sizes, large minimum leaf sizes, and recent-season weighting are sensible protections against overfitting. Using both short- and long-term player history is also useful.
One important documentation correction: the code does not explicitly project team volume, allocate it among players, and multiply opportunity by efficiency. It predicts player outcomes directly using those concepts as features. There are no constraints ensuring that player projections reconcile to team totals.
That does not make it a bad model. It means the documentation describes a more structured system than [the implementation actually provides (line 37)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/model.py:37).
What the historical results support
I reproduced the published report from its underlying prediction files. The comparison covers 62 distinct weeks across 2022–2025, with separate RB, WR, and TE evaluations.
These are the existing backtest’s pairwise accuracy figures: how often each source correctly orders two players who subsequently played and scored different point totals.
Position	Your model	Archived expert consensus	Difference
RB	63.15%	62.98%	+0.16 percentage points
WR	61.01%	61.95%	−0.93 points
TE	60.88%	59.99%	+0.90 points
Overall, equally weighting position-weeks	61.68%	61.64%	+0.04 points


The reasonable interpretation is roughly competitive in this retrospective experiment, with WR the weakest relative position.
An approximate bootstrap resampling whole weeks puts the overall difference around −0.50 to +0.59 percentage points. That interval does not account for all temporal dependence, model selection, or leakage, but it already shows that the apparent overall advantage is inconclusive.
There is evidence of value beyond a simple recent-performance average: using half-credit for prediction ties, that baseline achieves about 59.32%, versus the model’s 61.68%.
Two qualifications make these figures more useful:
- Difficult decisions are much harder. Restricting comparisons to players within five ECR ranking places, your model’s pairwise accuracy falls to approximately 52.6% for WR, 54.6% for RB, and 55.9% for TE. The broad 61.7% figure should not be interpreted as the probability of correctly answering a close lineup question.
- The headline points error includes many peripheral players. PPR mean absolute error is 3.77 points overall, but 6.00 points among players projected for at least eight. Neither figure proves good or bad performance without a matched benchmark; the second better represents many actual start/sit decisions.
The most important findings
1. Some historical features use postgame information.
   The historical “starting quarterback” is whichever player ultimately attempted the most passes. That can identify an injury replacement or someone who entered after a benching.
   I found 105 disagreements across 2,174 team-games between this selection and the schedule’s recorded starter in 2022–2025. This is a concrete information leak, although its net effect on accuracy has not been measured. [QB selection code (line 155)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/features.py:155)
   Teammate and defensive-back absences are also inferred from whether players actually appeared. “Took no snaps” is not equivalent to “known unavailable before kickoff.” Moreover, excluding inactive players from the scoring comparison does not remove the advantage of knowing which teammates and opponents ultimately played. [Absence features (line 315)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/features.py:315)
   Historical weather and betting lines also lack the timestamped snapshots needed to establish that they match the benchmark’s information cutoff. The backtest therefore does not yet demonstrate an equal-information contest.
2. The live product has not been evaluated as a complete system.
   The historical backtest evaluates the statistical forecast without the player-prop blend. The live pipeline applies that blend before generating ranges and probabilities.
   In the inspected snapshot, 331 of 345 players had some market input. This is a major part of the product, not an occasional adjustment.
   Consequently, “the model backtested near consensus” does not establish the accuracy of the numbers currently published. The market blend might improve them, but that remains a hypothesis. [Live pipeline (line 76)](/Users/avib/Documents/CloakandKernel/ff-projections/scripts/run_weekly.py:76)
3. The uncertainty modeling is promising, with a validation problem that is fixable.
   Training ranges on out-of-sample point predictions is a good choice.
   However, the existing evaluation trains each season’s range model on all other seasons. When evaluating 2022, that includes outcomes from 2023–2025. That is cross-validation across seasons, not a historical forecasting simulation. Chronological evaluation should train on earlier periods only. [Backtest implementation (line 65)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/backtest.py:65), scikit-learn’s time-series validation guidance
   I ran a cleaner check: train the range model on 2022–2024 and evaluate 2025. Among 1,874 players projected for at least eight PPR points, the nominal 80% interval contained 80.6% of outcomes. That is encouraging. The predicted median was somewhat low: 53.4% of outcomes fell below it.
   This supports the range-model concept. It does not validate the market-blended live distributions, every player subgroup, or head-to-head probabilities.
4. The backtest unfairly penalizes tied predictions.
   Its pairwise metric gives a tied forecast zero credit whenever the actual results differ. The rank-average ensemble generates more ties than either constituent forecast, so this convention particularly hurts the ensemble.
   Giving prediction ties half-credit changes the overall comparison to:
   Forecast	Pairwise accuracy
   Recent-performance baseline	59.32%
   Expert consensus	61.67%
   Your model	61.68%
   Model + consensus rank blend	62.02%
   
   
   That is not proof the blend will win prospectively. It does mean the existing report’s unfavorable assessment of blending is sensitive to its scoring convention. [Pairwise metric (line 37)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/backtest.py:37)
5. The market conversion makes meaningful assumptions.
   Credit where due: the code recognizes that an over/under line is not automatically an expected value. It uses both sides of an over/under market to remove the margin proportionally, then fits a distribution to infer a mean.
   The less-established pieces are:
   - Fixed yardage variability assumptions across players.
   - A Poisson assumption for receptions and touchdown counts.
   - A blanket 10% probability reduction for anytime-touchdown prices.
   - A fixed 50% blending weight without historical validation.
   One market probability does not uniquely determine a mean without distributional assumptions. These choices therefore need sensitivity checks and prospective evaluation. [Market conversion (line 205)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/live.py:205)
   There is also a questionable weighting side effect: merely obtaining a prop can change the forecast even when that prop exactly matches the model’s component forecast. My controlled check moved a half-PPR projection from 12 to 11 despite an unchanged receptions estimate. The blend changes the relative influence of the component and direct models as well as incorporating market information. [Blending code (line 72)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/publish.py:72)
6. The explanations do not fully explain the displayed number.
   Every factor effect is calculated in model-only half-PPR points. Those same effects are shown when the user selects PPR or standard, and when the published forecast incorporates betting markets.
   The factors also represent sensitivity to chosen hypothetical baselines. They are not causal estimates, and they do not sum into a complete explanation of the projection. [Explanation code (line 49)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/publish.py:49)
   This feature could become a strong differentiator. It needs clearer labeling, scoring-specific calculations, and an explicit explanation of the market adjustment.
7. The head-to-head recommendation answers an incomplete decision question.
   It independently samples two player distributions and recommends whoever more often outscores the other.
   Three separate objectives matter:
   - Highest expected fantasy points.
   - Highest probability of outscoring another player.
   - Highest probability that your entire lineup wins.
   These can favor different players. The current tool computes an approximation to the second and labels it “Start,” without your lineup or opponent context.
   Independence also misses shared-game and teammate relationships, and no head-to-head probability calibration is reported. Twenty thousand simulations reduce simulation noise; they do not establish that the assumed distributions are correct. [Comparison implementation (line 219)](/Users/avib/Documents/CloakandKernel/ff-projections/site/index.html:219)
How it compares with alternatives
Alternative	Where yours stands
Simple rolling-average projections	Your features and modeling are substantially richer. The available retrospective results show improvement, subject to the identified evaluation problems.
FantasyPros consensus	The available result is effectively a tie overall, with weaker WR performance. Your ranges and explanations add useful presentation, but superiority has not been established.
Professional projection systems	You use many relevant signals, but lack explicit team/player reconciliation and a demonstrated process for handling role changes, rookies, and unusual news.
Sportsbook-derived expectations	You incorporate their information thoughtfully, but do not independently establish an edge over it. The conversion and blending choices introduce their own modeling risk.


For a concrete professional comparison, Establish the Run publicly describes projecting team play volume and pass share, assigning player opportunity shares, estimating efficiency, and making selected manual adjustments. Its cited description concerns season-long projections, so it is an architectural reference rather than a direct weekly accuracy benchmark. ETR methodology
Fantasy Life describes weekly projections that account for team archetypes and game scripts, alongside rankings updated for breaking news. Your model contains related historical inputs, but does not explicitly simulate those scenarios or implement that kind of editorial news response. These are provider descriptions, not independent proof that their predictions are better. Fantasy Life’s product description
Also, your pairwise metric is different from FantasyPros’ official accuracy methodology. These results cannot establish where your model would rank among its named experts. FantasyPros methodology
I cannot credibly rank this against proprietary sportsbook models or declare a “best model” from the available public evidence.
How useful is the product today?
Its focused scope is a strength. Position filters, three scoring formats, search, expandable player details, and a quick comparison workflow fit the intended task. I tested the player-detail and comparison interactions successfully.
Several practical issues reduce trust:
- On narrow screens, the range and “Why” columns disappear, hiding two of the product’s strongest features until users expand a player.
- The inspected payload contained 25 weather explanations displaying nan°F, wind nan mph.
- A missing game-line timestamp is presented as “Game lines unavailable,” although player details still contain lines. Failure to verify freshness should be distinguished from missing data.
- If every weather request fails, the interface can label the week “Indoors.”
- The weekly history file is overwritten on each rebuild. Git preserves earlier versions, but the final weekly JSON is not a complete archive of each player’s forecast before kickoff. [History writer (line 185)](/Users/avib/Documents/CloakandKernel/ff-projections/ffmodel/publish.py:185)
The public data sources also update on different schedules. For example, nflverse documents separate timing for player statistics, snaps, and other feeds; a successful rebuild alone does not demonstrate that every input is current. nflverse update schedule
What would most improve the grade
I would prioritize these in order:
1. Repair the evaluation. Remove postgame QB and participation information; establish matched forecast cutoffs; evaluate distributions chronologically; handle ties explicitly.
2. Freeze forecasts before kickoff. Save each version, input timestamps, model version, scoring settings, availability assumptions, and market coverage. Keep model-only and blended forecasts separately.
3. Evaluate actual lineup decisions. Report close-choice accuracy, points lost from choosing the lower scorer, cross-position flex comparisons, and results by scoring format. Include players who were recommended but did not play.
4. Align explanations and recommendations with their meaning. Explain the displayed forecast, show the market adjustment, and distinguish projected points from head-to-head probability.
5. Then improve the model where errors concentrate. Test role-change handling, rookie priors, better usage signals, and team-level reconciliation. Run ablations to establish which additions help.
For prospective tracking, I would compare the model-only forecast, market blend, consensus, a simple baseline, and a preselected model-consensus blend at the same decision times. Measure point error, decision regret, interval coverage and width, and probability calibration. Quantile forecasts should also be assessed with a proper loss such as pinball loss, rather than coverage alone. Evaluation guidance
The strongest reason to keep developing this is that the foundation already produces plausible, reproducible forecasts and useful uncertainty ranges. The biggest opportunity is to make its evidence and product claims as rigorous as its modeling ambitions.
