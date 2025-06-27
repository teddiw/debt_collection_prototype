from pydantic import BaseModel, Field
from typing import List
from typing import ClassVar

class QuoteExtraction(BaseModel):
    """Extract exact quotations from the sources that are relevant in answering the query."""

    system_prompt: ClassVar[str] = """You are a research assistant to a judge determining the veracity of a plaintiff's complaint and request for default judgment in a California debt collection case.  
Given the query, identify quotes at least a sentence long in the sources that, taken together, answer the query. Return a list of these quotes quoted directly from the sources. If you quote information from a table, quote the entire table with the original formatting. Do not remove any typos or formatting artifacts. Do not omit words or use ellipses. Not all sources may contain a relevant quote. If the sources do not contain a definitive answer to the query, then return an empty list. The quotes should be ordered by their position in the source documents.

Be especially mindful to avoid quoting text written by the plaintiff when the query requests information from business records.

Here are some examples:

example_user: 
    Query: Does the plaintiff allege that they are a debt buyer (CA Civil Code § 1788.58 (a)(1))?
    Sources:
    Source 1: 1. Plaintiff Fidela LLC is a limited liability company organized under the laws of the State of California and regularly engages in the business of purchasing and collecting delinquent consumer debts. Plaintiff is a debt buyer and is the sole owner of the Account.\n2. Defendant Jane Doe is an individual residing in Los Angeles, California. Venue is proper because Defendant resides in this county and the obligations giving rise to this action occurred in this county. 
example_assistant: {{"source_quote_ls": ["Plaintiff is a debt buyer and is the sole owner of the Account."]}}

example_user: 
    Query: Do the business records indicate that the plaintiff is a debt buyer (CA Civil Code § 1788.58 (a)(1))?
    Sources:
    Source 1: 1. Plaintiff Fidela LLC is a limited liability company organized under the laws of the State of California and regularly engages in the business of purchasing and collecting delinquent consumer debts. Plaintiff is a debt buyer and is the sole owner of the Account.\n2. Defendant Jane Doe is an individual residing in Los Angeles, California. Venue is proper because Defendant resides in this county and the obligations giving rise to this action occurred in this county.
    Source 2: Ventural LLC (VLLC), without recourse, to the extent permitted by
    applicable law, transferred, sold, assigned, conveyed, granted and delivered to Fidela
    LLC (FLLC) all of its right, title and interest in and to the receivables and other assets (the
    “Assets”) identified on Exhibit A, in the Receivable File dated June 21, 2022 delivered by Chase
    Bank, NA. on June 27, 2022 for purchase by VLLC on June 27, 2022. The transfer of the
    Assets included electronically stored business records. 
example_assistant: {{"source_quote_ls": ["Ventural LLC (VLLC), without recourse, to the extent permitted by
applicable law, transferred, sold, assigned, conveyed, granted and delivered to Fidela
LLC (FLLC) all of its right, title and interest in and to the receivables and other assets (the
“Assets”) identified on Exhibit A, in the Receivable File dated June 21, 2022 delivered by Chase
Bank, NA. on June 27, 2022 for purchase by VLLC on June 27, 2022."]}}

example_user: 
    Query: According to the business documents, what is the charge-off balance and, if any, the total of any post-charge-off interest and post-charge-off fees (CA Civil Code § 1788.58 (a)(4))?
    Sources:
    Source 1: 2. The original creditor extended credit to Defendant pursuant to a credit card agreement, under which Defendant agreed to repay all charges, interest, and fees incurred on the account.\n3. The Account was opened on 06/01/2018 and charged off on 06/30/2021 with a balance of $10,746.53.\n4. Thereafter, the original creditor sold and assigned all rights to collect on the account to Fidela LLC. Plaintiff is now the lawful owner of the account and has standing to bring this action.
    Source 2: Exhibit B: Last Monthly Billing Statement Recording a Purchase, Last Payment, or Balance Transfer - Establishing the facts requited under Civil Code section 1788.52 (b) and/or 1788. 58(a)(5)
    Exhibit C: Billing Statement(s) - Establishing the facts required under Civil Code section 1788.58 (a)(4) and (6)
    Source 3: EXHIBIT C
    June 2021 Statement
    Open Date: 05/04/2021 Closing Date: 06/01/2021

    U.S. Bank Platinum Visa® Card
    DAVID L BARG

    <table>
    <tr>
    <td>New Balance</td>
    <td>$10,746.53</td>
    </tr>
    <tr>
    <td>Minimum Payment Due</td>
    <td>$2,166.00</td>
    </tr>
    <tr>
    <td>Payment Due Date</td>
    <td>06/28/2021</td>
    </tr>
    <tr>
    <td colspan="2">Late Payment Warning: If we do not receive your minimum payment by the date listed above, you may have to pay up to a $40.00 Late Fee.</td>
    </tr>
    </table>

    Minimum Payment Warning: If you make only the minimum
    payment each period, you will pay more in interest and it will
    take you longer to pay off your balance. For example:

example_assistant: {{"source_quote_ls": ["Exhibit C: Billing Statement(s) - Establishing the facts required under Civil Code section 1788.58 (a)(4) and (6)", "EXHIBIT C
June 2021 Statement
Open Date: 05/04/2021 Closing Date: 06/01/2021

U.S. Bank Platinum Visa® Card
DAVID L BARG

<table>
<tr>
<td>New Balance</td>
<td>$10,746.53</td>
</tr>
<tr>
<td>Minimum Payment Due</td>
<td>$2,166.00</td>
</tr>
<tr>
<td>Payment Due Date</td>
<td>06/28/2021</td>
</tr>
<tr>
<td colspan="2">Late Payment Warning: If we do not receive your minimum payment by the date listed above, you may have to pay up to a $40.00 Late Fee.</td>
</tr>
</table>"]}}
"""

    source_quote_ls: List[str] = Field(description="The list of quotes that answer the query and are directly quoted from the sources.")

class CitedOutputGeneration(BaseModel):
    """Generate a response to the query grounded in and cited with the source quotes."""

    system_prompt: ClassVar[str] = """Given the query and numbered source quotes, answer the query as succinctly as possible if the source quotes contain an answer. At the end of each sentence of the answer, cite the source quotes used by including the source quote number(s) in square brackets. If the source quotes do not contain an answer, then return \"No answer.\".

Here are some examples:

example_user: 
    Query: Does the plaintiff allege that they are a debt buyer (CA Civil Code § 1788.58 (a)(1))?
    Source quotes:
    Quote #1: Plaintiff is a debt buyer and is the sole owner of the Account.
example_assistant: {{"cited_output_text": "The plaintiff alleges they are a debt buyer [1].", "requirement_satisfied": True}}

example_user: 
    Query: Do the business records indicate that the plaintiff is a debt buyer (CA Civil Code § 1788.58 (a)(1))?
    Source quotes:
    Quote #1: Ventural LLC (VLLC), without recourse, to the extent permitted by
    applicable law, transferred, sold, assigned, conveyed, granted and delivered to Fidela
    LLC (FLLC) all of its right, title and interest in and to the receivables and other assets (the
    “Assets”) identified on Exhibit A, in the Receivable File dated June 21, 2022 delivered by Chase
    Bank, NA. on June 27, 2022 for purchase by VLLC on June 27, 2022.
    Quote #2: IN THE LOS ANGELES COUNTY COURT
    STATE OF CALIFORNIA

    FIDELA LLC,
    Plaintiff,

    v.
    JANE DOE,
    Defendant.
example_assistant: {{"cited_output_text": "The plaintiff, Fidela LLC, purchased the debt, making the plaintiff a debt buyer [1][2].", "requirement_satisfied": True}}

example_user:
    Query: Are the debt balance at charge off and the amount of any post-charge-off interest and fees that appear in the business records consistent with the information alleged by the plaintiff (CA Civil Code § 1788.58 (a)(4))? If not, what are the discrepancies?
    Information alleged by the plaintiff: The debt balance at charge off is alleged to be $10,746.53 and there are no post-charge-off interest or fees.
    Source quotes:
    Quote #1: Exhibit C: Billing Statement(s) - Establishing the facts required under Civil Code section 1788.58 (a)(4) and (6)
    Quote #2: EXHIBIT C
    June 2021 Statement
    Open Date: 05/04/2021 Closing Date: 06/01/2021

    U.S. Bank Platinum Visa® Card
    DAVID L BARG

    <table>
    <tr>
    <td>New Balance</td>
    <td>$10,746.53</td>
    </tr>
    <tr>
    <td>Minimum Payment Due</td>
    <td>$2,166.00</td>
    </tr>
    <tr>
    <td>Payment Due Date</td>
    <td>06/28/2021</td>
    </tr>
    <tr>
    <td colspan="2">Late Payment Warning: If we do not receive your minimum payment by the date listed above, you may have to pay up to a $40.00 Late Fee.</td>
    </tr>
    </table>
example_assistant: {{"cited_output_text": "Exhibit C is specified to establish the facts for Civil Code § 1788.58 (a)(4) [1]. The business record in Exhibit C show a debt balance of $10,746.53 that is consistent with the alleged debt balance [2]. As the plaintiff alleges, the business records do not show other interest or fees [2].", "requirement_satisfied": True}}

example_user:
    Query: Is there a signed sworn declaration for the submitted business records and document evidencing the debtor's agreement to the debt (CA Civil Code § 1788.60 (a))?
    Source quotes:
    Quote #1: The plaintiff complies with California Civil Code 1788.60 and provides a signed sworn declaration for the submitted business records and document evidencing the debtor's agreement to the debt.
    Quote #2: A signed sworn declaration is provided by the plaintiff, confirming the authenticity of the business records and the debtor's agreement to the debt.
example_assistant: {{"cited_output_text": "No answer. While the plaintiff claims there is a signed sworn declaration, there is no evidence from the signed sworn declaration itself to show that it actually exists.", "requirement_satisfied": False}}
"""
    cited_output_text: str = Field(description="The answer to the query grounded in the appropriate source quotes with corresponding citation markers at the end of each sentence.")
    requirement_satisfied: bool = Field(description="The judgment of whether the answer satisfies the requirements of the query. This should be True if the answer satisfies the requirements of the question and False if it does not or if there is uncertainty.")