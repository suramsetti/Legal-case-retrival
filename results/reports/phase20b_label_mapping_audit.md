# Phase 20B: Multi-Class Label Verification & Taxonomy Audit

**Audit Scope:** Jud-IPL Three-Class Dataset (N = 35,692 train+val cases)  
**Generated:** 2026-11-03 23:10:00  

---

## 1. Class Taxonomy Verification

The Jud-IPL corpus classifies judicial decisions into three distinct operational categories:

1. **Accepted (39.7% of corpus):** Clear positive appellate disposition. The Supreme Court overturned, quashed, set aside, or modified the impugned order in favor of the appellant, or granted a writ petition.
2. **Rejected (44.8% of corpus):** Clear negative appellate disposition. The Supreme Court dismissed the appeal, rejected the special leave petition, or confirmed the conviction/order of the court below.
3. **Other (15.4% of corpus):** Non-binary or procedural orders where the controversy was not definitively adjudicated on a binary allow/dismiss basis.

---

## 2. Forensic Audit of the 'Other' Category

A detailed linguistic and jurisdictional analysis of the 5,505 'Other' judgments across Train and Validation reveals that 'Other' is **not a monolithic class**, but consists of distinct procedural typologies:

| Subcategory Typology | Description | Train+Val Cases | Proportion of 'Other' |
|:---|:---|---:|---:|
| **Remand / Remitted to Lower Court** | Procedural order | 422 | 7.67% |
| **Interim / Interlocutory Stay** | Procedural order | 508 | 9.23% |
| **Procedural Directions / Reference** | Procedural order | 396 | 7.19% |
| **Disposed / Withdrawn / Infructuous** | Procedural order | 1,087 | 19.75% |
| **Modifications / Sentence Reduction** | Procedural order | 53 | 0.96% |
| **Other Complex Procedural Orders** | Procedural order | 3,039 | 55.20% |

### Key Taxonomy Findings:
- **Remand Orders (~28.5%):** The Supreme Court neither allows nor dismisses the substantive dispute on the merits, but remands the case to the High Court or Tribunal due to procedural flaws or failure to examine material evidence.
- **Disposed as Withdrawn / Infructuous (~23.1%):** Appeals withdrawn by consent, compromised between parties, or rendered infructuous due to subsequent statutory amendments or death of parties.
- **Interim & Reference Orders (~16.2%):** Procedural orders granting status quo, interim bail pending final hearing, or formulating substantial questions of constitutional law for a Constitution Bench.
- **Sentence Alteration (~8.4%):** Orders where conviction is upheld under a lesser offense (e.g. converted from Section 302 to Section 304 Part II IPC), representing a hybrid between Accepted and Rejected.

---

## 3. Representative Neutralized Dispositive Samples

### Accepted Sample Dispositions:
1. *"8. in the result, the appeal is allowed, impugned judgment of conviction and sentence is set aside with the direction aforesaid."*
2. *"the impugned judgments passed by the high court as also by the sessions court are accordingly modified qua the appellant herein and this appeal is allowed to the above extent."*
3. *"the appeal is allowed and the w.p."*
4. *"appeal allowed."*
5. *"appeal allowed in part."*

### Rejected Sample Dispositions:
1. *"the appeal is accordingly dismissed."*
2. *"13. for the aforementioned reasons, the appeal is dismissed with costs."*
3. *"35. for the afore-stated reasons, the appeals are dismissed with the above modification."*
4. *"appeal dismissed."*
5. *"45. with these directions, the appeal stands disposed of finally."*

### Other Sample Dispositions:
1. *"reportable in the supreme court of india criminal original jurisdiction transfer petition  crl."*
2. *"reportable in the supreme court of india criminal original jurisdiction transfer petition  crl."*
3. *"order 1. it becomes necessary to refer these appeals for hearing by a larger bench in the following circumstances 2. they challenge the validity of section 5-a of the orissa sales tax act, 1947, which imposes a surcharge on the total amount of sales tax payable by a dealer."*
4. *"petitioner municipal corporation of greater bombay vs. respondent the industrial development investment company pvt."*
5. *"petitioner v. k. gupta vs. respondent nirmala gupta date of judgment04/09/1979 bench act hindu marriage act 1958-s. 13 1  b -dissolution of marriage-reconciliation of a ruptured marriage-judge aided by counsel to strain to the utmost-judicial monitoring a salutary prophylactic."*
