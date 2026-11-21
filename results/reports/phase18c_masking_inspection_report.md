# Phase 18C: Dispositive Holding Masking Inspection Report

**Generated:** 2026-11-03 18:37:02

## Masking Pipeline

1. **Proof Sentence Removal:** Exact proof_sentence span replaced with [MASKED_HOLDING]
2. **Original Text Preservation:** Original full_text preserved separately
3. **Note:** Full masking will be applied during model training. This report shows sample results.

## Sample Statistics

**Binary Task (n=1000):**
- Proof sentences masked: 639 (63.9%)
- Average masking percentage: 0.66%
- Max masking percentage: 30.53%

**Three-Class Task (n=1000):**
- Proof sentences masked: 592 (59.2%)
- Average masking percentage: 0.57%
- Max masking percentage: 24.75%

## Sample Inspection

### Binary Task Samples

#### Sample 1: 0000206686 (Label: Rejected)

**Proof Sentence:** with these observations the appeal is disposed of....

**Original Text:** act headnote judgment o r d e r leave granted.
after hearing learned counsel for the parties, we are of the opinion that the interpretation placed by the learned division bench of the high court on section 104 of the motor vehicles act, 2988 is unexceptionable and calls for no interference.
we, howe...

**Masked Text:** act headnote judgment o r d e r leave granted.
after hearing learned counsel for the parties, we are of the opinion that the interpretation placed by the learned division bench of the high court on section 104 of the motor vehicles act, 2988 is unexceptionable and calls for no interference.
we, howe...

**Proof Masked:** True

#### Sample 2: 0100361579 (Label: Rejected)

**Proof Sentence:** the impugned judgment and order donot call for any interference.this appeal,therefore, fails andis dismissed,but in the peculiar facts of the case without any order asto costs....

**Original Text:** the case has a checquered history but we shall refer to the facts to the extentrelevant for the purpose of disposal of this appeal.
on 23rd july 1961 the predecessors of respondents filed a civil suit in thecourt of civil judge  seniordivision, bicholin  for declaration that they are the owners andp...

**Masked Text:** the case has a checquered history but we shall refer to the facts to the extentrelevant for the purpose of disposal of this appeal.
on 23rd july 1961 the predecessors of respondents filed a civil suit in thecourt of civil judge  seniordivision, bicholin  for declaration that they are the owners andp...

**Proof Masked:** True

#### Sample 3: 0001581571 (Label: Rejected)

**Proof Sentence:** appeal dismissed....

**Original Text:** legal representatives of a partner--if entitled to share in good-will of a firm continuing after death of a partner.
headnote d who was one of eight partners in a firm, died on february 20, 1957.
by virtue of a provision in the partnership deed, the business of the firm was continued by the survivin...

**Masked Text:** legal representatives of a partner--if entitled to share in good-will of a firm continuing after death of a partner.
headnote d who was one of eight partners in a firm, died on february 20, 1957.
by virtue of a provision in the partnership deed, the business of the firm was continued by the survivin...

**Proof Masked:** True

### Three-Class Task Samples

#### Sample 1: 0000186097 (Label: Accepted)

**Proof Sentence:** 13. accordingly, this appeal succeeds and is allowed, and the judgment and order of the letters patent appeal bench in first appeal number 165 of 1978 on the file of the high court of punjab and harya...

**Original Text:** ledged goods were insured.
it was admitted that the bank as the pledgee of the pledged goods received an amount of rs.
59,570/2/- from the insurer.
ordinarily, this would tantamount to payment of the same amount in cash credit account.
the pledged goods were of the firm.
they could have been sold an...

**Masked Text:** ledged goods were insured.
it was admitted that the bank as the pledgee of the pledged goods received an amount of rs.
59,570/2/- from the insurer.
ordinarily, this would tantamount to payment of the same amount in cash credit account.
the pledged goods were of the firm.
they could have been sold an...

**Proof Masked:** False

#### Sample 2: 0001909386 (Label: Accepted)

**Proof Sentence:** appeal is allowed accordingly....

**Original Text:** number4, application for substitution of the heirs of deceased original petitioner, is granted.
we have heard learned senior counsel for the appellants, heirs of the original petitioner in the s.l.p.
as well as learned senior counsel for the respondents finally.
a bench of two judges of this court h...

**Masked Text:** number4, application for substitution of the heirs of deceased original petitioner, is granted.
we have heard learned senior counsel for the appellants, heirs of the original petitioner in the s.l.p.
as well as learned senior counsel for the respondents finally.
a bench of two judges of this court h...

**Proof Masked:** False

#### Sample 3: 0001542360 (Label: Rejected)

**Proof Sentence:** 8. we, therefore, dispose of this appeal with the direction that the republic high school shall continue to be located in the disputed building until it is shifted in accordance with law....

**Original Text:** the appeal is directed against the judgment of the division bench of the andhra pradesh high court dated 11-7-1996, dismissing the appeal filed by the present appellant and confirming the order or directions issued by the learned single judge in writ petition number 17419 of 1995 as modified in revi...

**Masked Text:** the appeal is directed against the judgment of the division bench of the andhra pradesh high court dated 11-7-1996, dismissing the appeal filed by the present appellant and confirming the order or directions issued by the learned single judge in writ petition number 17419 of 1995 as modified in revi...

**Proof Masked:** False

