# Background Screening Industry Newsletter Sources: A Technical Inventory

The background screening industry requires monitoring **247+ news sources** across regulatory, legal, and industry channels spanning six continents. This research identifies high-value sources with RSS feed availability, authentication requirements, and content relevance for building a newsletter aggregation proof-of-concept. **JDSupra, Lexology, and major government RSS feeds provide the most accessible automated content**, while critical industry sources like PBSA and many data protection authorities require scraping or manual monitoring.

---

## Primary sources for automated monitoring

These sources offer high relevance to background screening compliance, regular updates, and accessible RSS feeds or APIs, making them ideal for automated newsletter aggregation.

### Legal and compliance publications

| Source | URL | RSS Feed | Auth | Update Freq | Relevance |
|--------|-----|----------|------|-------------|-----------|
| **JDSupra** | jdsupra.com | ✅ `jdsupra.com/resources/syndication/docsRSSfeed.aspx?ftype=LaborEmploymentLaw&premium=1` | Free | Daily | Very High - FCRA, ban-the-box, employment screening |
| **Lexology** | lexology.com | ✅ Custom feeds via `/account/rss` | Free registration | 450+/day | High - 170+ jurisdictions, employment law |
| **National Law Review** | natlawreview.com | ✅ `natlawreview.com/recent-contributions/feed` | Free | Daily | High - employment, privacy law |
| **HR Dive** | hrdive.com | ✅ `hrdive.com/feeds/news/` | Free | Daily | High - background check settlements, compliance |
| **Fisher Phillips** | fisherphillips.com | ✅ `/news-insights/rss.xml` | Free | Weekly | High - dedicated FCRA practice |
| **Consumer Finance Monitor** | consumerfinancemonitor.com | ✅ `/feed/` | Free | Daily | High - CFPB, FCRA tracking |

### US Federal regulatory sources

| Source | URL | RSS Feed | Focus |
|--------|-----|----------|-------|
| **Federal Register** | federalregister.gov | ✅ Agency-specific and custom search RSS | Employment rulemaking, FCRA |
| **CFPB Blog** | consumerfinance.gov | ✅ `/about-us/blog/feed/` | FCRA enforcement, advisory opinions |
| **FTC Press Releases** | ftc.gov | ✅ `/news-events/news/press-releases/rss.xml` | Background screening guidance |
| **EEOC Newsroom** | eeoc.gov | ✅ `/newsroom/rss-feeds` | Criminal background check discrimination |
| **DOL News** | dol.gov | ✅ `/rss/releases.xml` | Labor law, OFCCP compliance |

### Background screening company blogs

| Source | URL | RSS Feed | Update Freq |
|--------|-----|----------|-------------|
| **First Advantage** | fadv.com/resources/blog/ | ✅ `fadv.com/feed` | 2 posts/month |
| **Sterling** | sterlingcheck.com/blog/ | ✅ `sterlingcheck.com/feed` | 3 posts/month |
| **HireRight** | hireright.com/blog | ✅ `/blog/feed` | Weekly |
| **National Student Clearinghouse** | studentclearinghouse.org | ✅ `/nscblog/feed/` | Regular |

### EU-wide regulatory sources

| Source | URL | RSS Feed | Languages |
|--------|-----|----------|-----------|
| **EDPB News** | edpb.europa.eu | ✅ `edpb.europa.eu/feed/news_en` | All 24 EU languages |
| **EUR-Lex** | eur-lex.europa.eu | ✅ Custom alerts (registration) | All EU languages |
| **EFDPO** | efdpo.eu | ✅ `/category/rss-feed/` | English |
| **GDPRhub** | gdprhub.eu | Manual | English - searchable case law |

---

## Secondary sources with occasional relevant content

These sources publish background screening content periodically or cover adjacent topics. Worth monitoring but lower priority for initial PoC.

### HR publications

| Source | URL | RSS Available | Auth | Notes |
|--------|-----|---------------|------|-------|
| **SHRM** | shrm.org | ✅ Chapter-level RSS | Mixed (free news/$299 membership) | High HR coverage, membership for tools |
| **HRM Asia** | hrmasia.com | ✅ Yes | Free | APAC focus |
| **HR in Asia** | hrinasia.com | ✅ Yes | Free | Regional HR news |
| **People Management (CIPD)** | peoplemanagement.co.uk | ✅ Yes | Some membership | UK employment law |
| **Canadian HR Reporter** | hrreporter.com | Newsletter | Partial subscription | Canadian context |
| **HR Executive** | hrexecutive.com | ✅ Yes | Free | HR tech, talent acquisition |

### Legal publications with screening coverage

| Source | URL | RSS | Auth | Notes |
|--------|-----|-----|------|-------|
| **Law360** | law360.com | ✅ Subscribers only | **Paid subscription** | 200+ articles/day, employment litigation |
| **Littler Mendelson** | littler.com | ✅ Yes | Free | Leading FCRA litigation practice |
| **Jackson Lewis** | jacksonlewis.com | ✅ Yes | Free | EEOC enforcement trends |
| **Ogletree Deakins** | ogletree.com | ✅ Yes | Free | CFPB, AI/algorithmic guidance |
| **Employment Law World View** | employmentlawworldview.com | Manual | Free | International employment |

### Credential and immigration sources

| Source | URL | RSS | Focus |
|--------|-----|-----|-------|
| **WES** | wes.org | Manual | International credential evaluation |
| **E-Verify** | e-verify.gov | Manual | Form I-9, employment eligibility |
| **USCIS I-9 Central** | uscis.gov/i-9-central | Manual | I-9 updates (Jan 2025 revision) |
| **NDASA** | ndasa.com | Manual | Drug testing compliance (absorbed DATIA) |
| **SAMHSA** | samhsa.gov | Manual | Federal drug testing (fentanyl added July 2025) |

---

## Regional sources by geography

### North America and Caribbean

**United States - State Level**
| Source | URL | Content | RSS |
|--------|-----|---------|-----|
| **NELP Ban the Box** | nelp.org | State/local law tracking (37 states + DC) | ❌ Email only |
| **California CRD** | calcivilrights.ca.gov | Fair Chance Act enforcement | ❌ Manual |
| **Accurate Ban the Box Map** | accurate.com/ban-the-box/ | Interactive compliance tool | ❌ Manual |
| **iProspectCheck** | iprospectcheck.com/ban-the-box-laws/ | State-by-state analysis | ❌ Manual |

**Canada**
| Source | URL | RSS | Notes |
|--------|-----|-----|-------|
| **OPC (Federal)** | priv.gc.ca | ✅ Multiple feeds at `/rss-feeds/` | PIPEDA enforcement, Certn investigation |
| **IPC Ontario** | ipc.on.ca | Manual | Police Record Checks Reform Act |
| **OIPC British Columbia** | oipc.bc.ca | Manual | Joint Certn investigation |
| **OIPC Alberta** | oipc.ab.ca | Manual | PIPA enforcement |
| **CAI Quebec** | cai.gouv.qc.ca | Manual | French primary, strictest biometrics |

**Caribbean**
| Jurisdiction | DPA Status | Notes |
|--------------|------------|-------|
| **Jamaica** | In force Dec 2023 | Data Protection Act 2020, GDPR-aligned |
| **Barbados** | Active | Data Protection Act 2019, "fully aligned" with GDPR |
| **Trinidad & Tobago** | NOT implemented | 2011 Act passed, no commissioner |

### Europe

**UK Sources**
| Source | URL | RSS | Relevance |
|--------|-----|-----|-----------|
| **ICO** | ico.org.uk | ⚠️ Under redesign | UK GDPR, employment data |
| **DBS** | gov.uk/government/organisations/disclosure-and-barring-service | ✅ GOV.UK RSS | **Critical** - criminal record checks |
| **GOV.UK Employment** | gov.uk/browse/employing-people | ✅ Topic RSS | Right to work, statutory requirements |
| **Home Office** | gov.uk | ✅ GOV.UK RSS | Immigration (£60K illegal worker fines) |
| **CIPD** | cipd.org | ✅ Via People Management | Employment Rights Act 2025 tracking |
| **Legal Island** | legal-island.ie | ❌ Subscription | Ireland/NI employment law |

**EU Data Protection Authorities**
| Country | DPA | RSS | Notes |
|---------|-----|-----|-------|
| **Ireland** | dataprotection.ie | ❌ Manual | Lead authority for Meta, TikTok, LinkedIn |
| **France (CNIL)** | cnil.fr | ✅ Aggregator available | Active enforcement, AI guidance |
| **Germany (BfDI)** | bfdi.bund.de | ❌ Manual | Federal; 16 state DPAs for private sector |
| **Netherlands (AP)** | autoriteitpersoonsgegevens.nl | ❌ Manual | Web scraping guidance May 2025 |
| **Spain (AEPD)** | aepd.es | ❌ Manual | Active enforcement |

**Germany Criminal Records**
- **Bundesamt für Justiz**: bundesjustizamt.de - Manages Führungszeugnis (certificate of conduct)
- New online apostille process via BfAA portal (2025)

**Nordic Countries** - All lack RSS; manual monitoring required
- Sweden (IMY): AI hiring tools investigation ongoing
- Denmark (Datatilsynet): Publishes audit focus bi-annually
- Finland: €1.1M pharmacy fine May 2025
- Norway: EEA member, GDPR via EEA Agreement

### Asia Pacific

**Priority Markets**
| Country | DPA/Source | RSS | Key Notes |
|---------|------------|-----|-----------|
| **Singapore PDPC** | pdpc.gov.sg | ✅ Confirmed | AI guidance, PET Sandbox |
| **Singapore MOM** | mom.gov.sg | ❌ Email alerts | Work Permit changes July 2025 |
| **Australia OAIC** | oaic.gov.au | ❌ Monthly newsletter | Compliance sweep Jan 2026 |
| **Australia Fair Work** | fairwork.gov.au | ✅ News feed | Baby Priya's Act Nov 2025 |
| **India (DPDP)** | meity.gov.in | ❌ Manual | **Critical** - DPDP Rules Nov 2025, full compliance May 2027 |
| **Japan PPC** | ppc.go.jp/en | ❌ Manual | Cross-border HR data guidance |
| **Hong Kong PCPD** | pcpd.org.hk | ❌ Manual | AI framework June 2024, proposed breach notification |
| **Philippines NPC** | privacy.gov.ph | ❌ Social media | Privacy Mark certification |
| **South Korea PIPC** | pipc.go.kr | ❌ Manual | KRW 15.1B record fines 2024, EU adequacy Sept 2025 |
| **China (via briefing)** | china-briefing.com | ✅ Yes | PIPL employment compliance |

**Regional HR Publications**
| Source | RSS | Coverage |
|--------|-----|----------|
| Singapore Law Watch | ✅ Yes | Legal news syndication |
| HRM Asia | ✅ Yes | Regional HR news |
| HR in Asia | ✅ Yes | APAC employment |
| Asia Law Portal | ✅ Yes | Legal market news |
| International Employment Lawyer | ✅ Yes | APAC Summit coverage |

### Africa and Middle East

**South Africa**
| Source | URL | RSS | Notes |
|--------|-----|-----|-------|
| **Information Regulator** | inforegulator.org.za | ❌ Manual | POPIA enforcement |
| **Department of Employment** | labour.gov.za | ❌ Manual | Official labor law |
| **Cliffe Dekker Hofmeyr** | cliffedekkerhofmeyr.com | Newsletter | Employment/POPIA alerts |
| **Mondaq South Africa** | mondaq.com/southafrica | ✅ Yes | Multi-firm legal updates |

**UAE/Gulf States**
| Source | URL | RSS | Notes |
|--------|-----|-----|-------|
| **MOHRE UAE** | mohre.gov.ae | ❌ Social media | Primary UAE labor authority |
| **Qiwa (Saudi)** | qiwa.sa | ❌ Registration | Mandatory contract registration |
| **Morgan Lewis Shifting Sands** | morganlewis.com/blogs/shiftingsandsoflaborlaw | ✅ Yes | **Best ME coverage** |
| **Gulf News** | gulfnews.com | ✅ Yes | Regional business news |
| **Arabian Business** | arabianbusiness.com | ✅ Yes | Employment trends |

**Other Markets**
| Country | Key Source | Notes |
|---------|------------|-------|
| **Nigeria** | NDPC (ndpc.gov.ng) | NDPA enforcement |
| **Kenya** | ODPC (odpc.go.ke) | Data Protection Act 2019 |
| **Israel** | PPA | Amendment 13 effective Aug 2025, fines up to NIS 640K |
| **Egypt** | EY, Clyde & Co alerts | New Labor Law No. 14/2025 effective Sept 2025 |

### South America

**Major Markets**
| Country | DPA | RSS | Key Developments |
|---------|-----|-----|------------------|
| **Brazil ANPD** | gov.br/anpd | ❌ Manual | Independent agency Sept 2025, SCCs Aug 2025, EU adequacy draft |
| **Argentina AAIP** | argentina.gob.ar/aaip | ❌ Manual | EU adequacy, Convention 108+ |
| **Colombia SIC** | sic.gov.co | ❌ Manual | Working week: 46→42 hours by 2026 |
| **Chile (New DPA)** | TBD (Dec 2026) | N/A | Law 21.719, fines up to 4% turnover |
| **Mexico (INAI→Ministry)** | N/A | ❌ Manual | INAI extinguished March 2025 |
| **Peru** | minjus.gob.pe | ❌ Manual | New regulation March 2025, DPO mandatory |

**Regional Publications**
| Source | RSS | Notes |
|--------|-----|-------|
| Baker McKenzie Employer Report | ✅ Yes | LATAM employment roundup |
| IAPP Brazil | ✅ Newsletter | LGPD/ANPD analysis |
| DataGuidance | Subscription | Comprehensive |
| ICLG Guides | ❌ Annual | Employment/data protection |
| TST (Brazil) | ✅ Press RSS | Supreme Labor Court decisions |

---

## Sources requiring authentication or subscription

### Paid subscriptions required

| Source | Cost | Value Proposition |
|--------|------|-------------------|
| **Law360** | Enterprise pricing | 200+ daily articles, employment litigation tracking |
| **DataGuidance** | Subscription | Comprehensive regulatory database, 170+ jurisdictions |
| **SHRM Full** | $299/year | Templates, toolkits, Ask an Advisor |
| **Legal Island** | Subscription | Ireland/NI employment law hub |
| **Lexology PRO** | Enterprise | API access, workflow tools |

### Free registration required

| Source | Registration Type | Benefits |
|--------|-------------------|----------|
| **Lexology** | Free account | Custom RSS feeds, 450+ daily articles |
| **Singapore Law Watch** | Free | Daily legal news |
| **EUR-Lex** | Free MyEUR-Lex | Custom RSS alerts |
| **EEOC** | GovDelivery | Email alerts |
| **California CRD** | Account | Complaint filing access |

### Membership-gated content

| Source | Membership | Gated Content |
|--------|------------|---------------|
| **PBSA** | Tiered by company size | Practical guidance, webinar library, member advisories |
| **IAPP** | Professional | Full news archive, training materials |
| **NDASA** | Membership | Drug testing certifications, training |
| **CIPD** | Professional | UK employment law toolkits |

---

## Technical implementation notes

### Sources with confirmed RSS feeds (priority for automation)

**Tier 1 - High-volume, high-relevance RSS**
```
JDSupra Labor/Employment: jdsupra.com/resources/syndication/docsRSSfeed.aspx?ftype=LaborEmploymentLaw&premium=1
Lexology: lexology.com/account/rss (custom, requires free account)
National Law Review: natlawreview.com/recent-contributions/feed
HR Dive: hrdive.com/feeds/news/
EDPB News: edpb.europa.eu/feed/news_en
Federal Register: federalregister.gov (agency-specific feeds)
CFPB Blog: consumerfinance.gov/about-us/blog/feed/
FTC Press: ftc.gov/news-events/news/press-releases/rss.xml
EEOC: eeoc.gov/newsroom/rss-feeds
DOL: dol.gov/rss/releases.xml
```

**Tier 2 - Company and industry blogs**
```
First Advantage: fadv.com/feed
Sterling: sterlingcheck.com/feed
HireRight: hireright.com/blog/feed
NSC: studentclearinghouse.org/nscblog/feed/
OPC Canada: priv.gc.ca/en/opc-news/rss-feeds/
Singapore PDPC: pdpc.gov.sg/RSS-Feeds
China Briefing: china-briefing.com (RSS available)
Morgan Lewis ME: morganlewis.com/blogs/shiftingsandsoflaborlaw
```

### Sources requiring web scraping

Most data protection authorities globally lack RSS feeds. Scraping approaches:

**Regular HTML scraping needed**
- PBSA (thepbsa.org/news/) - News page, last public items 2019
- Ireland DPC (dataprotection.ie/en/news-media/latest-news)
- UK ICO (ico.org.uk) - RSS temporarily unavailable during redesign
- Australia OAIC (oaic.gov.au)
- Hong Kong PCPD (pcpd.org.hk)
- Philippines NPC (privacy.gov.ph)
- All South American DPAs (ANPD, AAIP, SIC)
- Germany BfDI (bfdi.bund.de)
- Netherlands AP (autoriteitpersoonsgegevens.nl)
- Nordic DPAs (IMY, Datatilsynet, etc.)
- E-Verify/USCIS updates
- NELP Ban the Box tracking

**Potential alternatives to scraping**
- Email newsletter subscriptions → Parse incoming emails
- Social media APIs (Twitter/X, LinkedIn) for DPA announcements
- Google Alerts for specific regulatory terms
- Third-party aggregators (Feedly, Inoreader with web-to-RSS)

### Content freshness by source type

| Source Type | Typical Update Frequency |
|-------------|-------------------------|
| Legal publications (JDSupra, Lexology) | 100-450 articles/day |
| Government regulatory | As events occur (weekly avg) |
| DPAs | Monthly guidance, irregular enforcement |
| Company blogs | 2-4 posts/month |
| Annual guides (ICLG) | Yearly editions |
| Association newsletters | Weekly (PBSA), monthly (OAIC) |

### Language considerations by region

| Region | Primary Languages | English Coverage |
|--------|-------------------|------------------|
| North America | English, French (Quebec) | ~95% |
| Europe | 24 EU languages + English | ~60% through international firms |
| Asia Pacific | English, Japanese, Chinese, Korean | ~70% via English publications |
| South America | Portuguese, Spanish | ~60% via international firms |
| Middle East | English, Arabic | ~80% business content in English |
| Africa | English, French, Arabic | ~75% English for major markets |

---

## Key industry developments affecting monitoring priorities

**Regulatory changes requiring tracking**
- **US**: CFPB advisory opinions on AI/algorithmic scoring (Circular 2024-06), FCRA litigation surge (+125% since 2014)
- **UK**: Employment Rights Act 2025, immigration fee increases (+32% Dec 2025), B2 English requirement Jan 2026
- **EU**: Digital Omnibus discussions, Brazil adequacy opinion Nov 2025
- **India**: DPDP Rules November 2025, full compliance May 2027
- **Chile**: New DPA effective December 2026
- **Brazil**: ANPD independence Sept 2025, EU adequacy draft
- **Mexico**: INAI abolished March 2025, functions to Ministry of Anti-Corruption

**Industry consolidation**
- First Advantage acquired Sterling ($2.2B, Oct 2024) - combine blog monitoring
- Accurate Background acquired Orange Tree (June 2024)
- Checkr acquired GoodHire (2022)
- DATIA merged into NDASA (June 2023) - update drug testing monitoring

**Ban the Box expansion**
- 37 states + DC have enacted laws
- 15 states extend to private employers
- 150+ cities/counties with local ordinances
- 80%+ US population covered by some form of fair chance policy

---

## Recommended implementation approach

**Phase 1: Core RSS aggregation (20 feeds)**
Focus on confirmed high-volume RSS feeds from legal publications, US federal agencies, and EDPB. These provide 500+ daily articles with minimal technical complexity.

**Phase 2: Add company and regional feeds (15 feeds)**
Integrate background screening company blogs and reliable regional sources with confirmed RSS (Morgan Lewis, China Briefing, Singapore PDPC, OPC Canada).

**Phase 3: Scraping infrastructure (25+ sources)**
Build scrapers for critical sources without RSS: PBSA, DPAs globally, state-level Ban the Box trackers, E-Verify updates.

**Phase 4: Email and social integration**
Subscribe to newsletters from sources without feeds (OAIC, Nordic DPAs, NELP) and parse for content. Monitor DPA social media accounts.

**Ongoing: Annual and quarterly sources**
Calendar-based collection of ICLG guides, HireRight Global Benchmark Report, PBSA conference materials, DPA annual reports.
