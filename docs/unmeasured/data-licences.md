# Data licences and terms — The Unmeasured Remainder

Checked 2026-11-14. This is a reading of published terms, not legal advice.
Quotes are short and taken verbatim from the pages named.

## 1. Transfermarkt datasets (dcaribou/transfermarkt-datasets)

**What applies**

- The GitHub repository declares **CC0-1.0**. The GitHub licence API reports
  `cc0-1.0`, and the `LICENSE` file is the CC0 1.0 Universal legal code.
  - https://github.com/dcaribou/transfermarkt-datasets
  - https://github.com/dcaribou/transfermarkt-datasets/blob/master/LICENSE
- The README describes the data as "built from Transfermarkt data". The
  repository's pipelines include a scraper.
- The README links to the Kaggle mirror
  (https://www.kaggle.com/datasets/davidcariboo/player-scores). Its licence
  field could not be read by a script (JavaScript page), so it is **not
  verified**. The GitHub licence is the one we rely on.

**Upstream caveat**

CC0 waives only the rights the maintainer holds. The underlying facts and
compilation come from transfermarkt.com, whose own terms could not be fetched
(the site refuses scripted access, and we do not work around that).

The only evidence found is second-hand and dated. On the R package-devel
mailing list (13 Feb 2018), a researcher quoted a reply from Transfermarkt:
the data is "copyrighted under German law" with "no license whatsoever", and
reproduction or online publication needs Transfermarkt's prior written
consent. The same reply said Transfermarkt supplies data sets to researchers
on request.
https://stat.ethz.ch/pipermail/r-package-devel/2018q1/002405.html

**What the paper and repo must do**

- **Cite** the dataset with its URL, the snapshot date (2026-09-05) and its
  CC0 licence, and **state that the data originate from Transfermarkt**
  (transfermarkt.com). Key: `transfermarktdatasets`.
- **Do not redistribute raw Transfermarkt-derived files** from this
  repository. `/data/` is already gitignored. Publish only derived aggregates
  (model outputs, ranks, summary tables).
- This is a conservative reading. CC0 would allow redistribution, but the
  upstream position is unresolved.
- Market values are a crowd estimate, not a fact; say so where they enter M6
  (`herm2014crowd`).

## 2. StatsBomb open data

**What applies**

- Repository: https://github.com/statsbomb/open-data (GitHub now also serves
  it as https://github.com/hudl/open-data).
- README "Terms & Conditions": "If you publish, share or distribute any
  research, analysis or insights based on this data, please state the data
  source as StatsBomb and use our logo". The logo is in the Media Pack:
  https://statsbomb.com/media-pack/
- `LICENSE.pdf` in the repository is the **StatsBomb Public Data User
  Agreement**, "last updated 8 September 2023":
  https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf
- Relevant clauses (paraphrased; short quotes marked):
  - 1.1: the data is provided for analysis and research.
  - 1.2.1: the user may not "edit, distort, distribute, reproduce, sell or in
    any way provide the data to any external or third party".
  - 1.2.2: no commercial exploitation of the data **or of analysis derived
    from it**.
  - 1.2.4: no use to publish material that might be defamatory or damaging
    to any individual or organisation.
  - 1.4: "The User is required to accredit any publication of analysis formed
    from StatsBomb Data with the StatsBomb brand logo."
  - 2.2: StatsBomb *asks* users to register a name and email at
    www.statsbomb.com/resource-centre before access.
  - 7: all data provided is StatsBomb's property; no redistribution without
    written consent.
  - 9: English law.

**What the paper and repo must do**

- **Print the StatsBomb logo** with any figure or table built from SB data,
  or once in the data section, which clause 1.4 requires for "any
  publication". **State "Data source: StatsBomb"** and cite
  `statsbombopendata`. The logo file must come from the Media Pack, and the
  logo is not committed to the repo until someone fetches it from there.
- **Never commit or publish raw SB events** or per-event derived files that
  reproduce them. Aggregates (player-season xT, xG correlations) are
  "analysis" and may be shared.
- **Non-commercial only.** This repository began as a betting app
  (`betting-app-yami`). Any commercial or betting use of results derived from
  SB data would breach 1.2.2.
- **Registration (clause 2.2)** is a request tied to personal data. It is
  Liam's decision whether to register. The agent did not do it.
- **Coverage caveat** (verified from the open-data match lists): inside the
  window, the SB club seasons are single-team samples (Barcelona 2019/20 and
  2020/21, PSG 2021/22 and 2022/23, Leverkusen 2023/24, 6 Inter Miami
  matches in 2023). The tournaments (WC 2022, Euro 2020 and 2024, Copa 2024,
  AFCON 2023) are complete. The paper must describe SB-based results as coming
  from that sample.

## 3. ESPN site API (site.api.espn.com / sports.core.api.espn.com)

**What applies**

- **No API terms exist.** ESPN publishes no documentation, licence or terms
  for these endpoints. Secondary sources say its public developer programme
  closed; they disagree on the year (2014 vs 2018), and no primary ESPN
  source was found. The endpoints serve no `robots.txt`
  (`site.api.espn.com/robots.txt` returns "Access Denied";
  `sports.core.api.espn.com/robots.txt` returns 403).
- **The general website terms do apply.** ESPN is a Disney property, and the
  **Disney Terms of Use** (https://disneytermsofuse.com/english/, "Last
  Updated: May 24, 2024") list ESPN among covered brands. As read on
  2026-11-14:
  - **2.A**: a licence "for your personal, noncommercial use only".
  - **2.B**: users may not "access, monitor, copy or extract the Disney
    Products using a robot, spider, script, or other automated means". This
    explicitly includes data mining and scraping, with an exception only for
    public search-engine indexing.
  - No copying except as expressly permitted.
- **www.espn.com/robots.txt** disallows, for all user agents,
  `/soccer/commentary`, `/soccer/matchstats` and `/soccer/lineups`. It
  disallows several AI crawlers entirely, including `anthropic-ai`.
  https://www.espn.com/robots.txt

**What this means**

The ESPN pipeline in design §3–§4.2 (scripted, cached download of match
commentary and rosters) is **in tension with the terms**: the Disney ToS
prohibit automated extraction, and the HTML commentary pages are disallowed
for crawlers. The commentary text also looks like syndicated Opta-style feed
text, so third-party rights may apply to the raw strings.

**Decision (Liam, 2026-11-15): option 1, ESPN dropped** (pre-registration
A15). The ESPN code, the 5,217 cached files and the processed `espn_*` tables
were deleted; nothing from ESPN enters the paper.

The options that were considered, from most to least conservative:

1. Drop ESPN. Zone-xG, zone-xA, fouls won and offsides then come only from
   StatsBomb (a narrow sample). H8 and the zone-xG analyses would need
   rescoping.
2. Ask ESPN/Disney for permission.
3. Proceed for non-commercial research. Keep the cache private, never
   publish raw commentary text or per-match ESPN tables, publish only
   aggregates, and state the source and its terms openly in the paper.

**If option 3 is taken, the paper must print:** that shot-level inputs were
collected from ESPN's public match pages and API for non-commercial research,
that no ESPN content is redistributed, and that ESPN/Disney terms restrict
automated access.

## 4. Data statement to print in the paper (draft)

> Match, lineup, event and market-value data: transfermarkt-datasets
> (github.com/dcaribou/transfermarkt-datasets, CC0-1.0, snapshot 2026-09-05),
> derived from Transfermarkt (transfermarkt.com). Event data with locations:
> StatsBomb open data (github.com/statsbomb/open-data), used under the
> StatsBomb Public Data User Agreement; [StatsBomb logo]. Ballon d'Or
> results: compiled by hand from the sources listed in
> paper-unmeasured/data/. No raw third-party data is redistributed with this
> paper; the code reproduces every number from the public sources.

## 5. American Soccer Analysis API (pre-registration A20 vi)

Read 2026-11-18. Used only for the coverage-fill variant.

**What applies**

- API: https://app.americansocceranalysis.com/api/v1, documented at
  https://app.americansocceranalysis.com/api/v1/__docs__/ (OpenAPI file:
  https://app.americansocceranalysis.com/api/v1/openapi.json). Its description
  reads: "Data returned by this API is licensed under CC BY-SA 4.0", with a
  link to https://spdx.org/licenses/CC-BY-SA-4.0.html.
- The README of ASA's client says: "The API is **free and public** — no
  authentication required."
  https://github.com/American-Soccer-Analysis/itscalledsoccer
- The README of ASA's web application asks: "If you use any of the data we
  make publicly available, we kindly ask that you provide credit and link back
  to the American Soccer Analysis website", and repeats "Data is licensed under
  CC BY-SA 4.0." https://github.com/American-Soccer-Analysis/asa-shiny-app
- CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Attribution:
  "You must give appropriate credit, provide a link to the license, and
  indicate if changes were made." ShareAlike: "If you remix, transform, or
  build upon the material, you must distribute your contributions under the
  same license as the original."
- No terms-of-use page, rate limit or `robots.txt` was found for the API host
  (`app.americansocceranalysis.com/robots.txt` returns 404). The documentation
  lists the query parameters, not the response fields; the fields were read
  from the responses. The pages read do not name ASA's upstream data
  provider, and CC BY-SA covers only the rights ASA holds.

**What is used**

- Requests, all with a generic User-Agent: the documentation page and its
  OpenAPI file, two small test calls to see the response fields (not kept),
  and 7 data requests sent once and cached under `data/players/raw/asa/`
  (`manifest.json` has each URL, time and SHA-256).
- `mls/players` (name and birth date, to match pool players), `mls/teams`,
  `mls/players/xgoals` (regular season; `minutes_played`, `goals`,
  `primary_assists`, and `goals` with `shot_pattern=Penalty`), `mls/games`
  (date and `expanded_minutes` of 107 games).
- Output: `data/players/processed/asa_mls_pool.csv.gz`, 5 player-seasons
  (Messi 2023 and 2024, Suárez 2024, Chiellini 2022 and 2023). The 2025 rows
  are downloaded only to compare ASA with the panel.

**What the paper and repo must do**

- **Credit ASA, link to it and to the licence, and say what was changed**
  (season totals taken as served; minutes also rescaled to a 90-minute match).
- **Share-alike** (conservative reading): a table or file we publish that
  contains these ASA values, or values computed from them, carries CC BY-SA
  4.0. `/data/` is gitignored, so nothing is redistributed today.
- Sentence to add to the §4 statement when the variant is reported:

> MLS regular-season minutes, goals, penalty goals and primary assists for
> 2022–2024: American Soccer Analysis (americansocceranalysis.com), from its
> public API (app.americansocceranalysis.com/api/v1), licensed CC BY-SA 4.0
> (creativecommons.org/licenses/by-sa/4.0/); retrieved 2026-11-18. [If
> `minutes_scaled` is used: minutes rescaled by us to a 90-minute match.]
