from pydantic import BaseModel, Field
from typing import List
from typing import ClassVar

class QuoteExtraction(BaseModel):
    """Extract exact quotations from the sources that are relevant in answering the query."""

    system_prompt: ClassVar[str] = """Given the query, identify quotes at least a sentence long in the sources that, taken together, answer the query. Return a list of these quotes quoted directly from the sources. If you quote information from a table, quote the entire table with the original formatting. Do not remove any typos or formatting artifacts. Do not omit words or use ellipses. Not all sources may contain a relevant quote. If the sources do not contain a definitive answer to the query, then return an empty list. The quotes should be ordered by their position in the source documents.

Here are some examples:

example_user: 
    Query: What is the debtor's name?
    Sources:
    Source 1: The original complaint is as follows: 1. The debtor's name is  John Doe. 2. The debtor's address is 123 Main St.
    Source 2: Big Bank is the charge-off creditor. They originally owned the debt.
example_assistant: {{"source_quote_ls": ["The debtor's name is  John Doe."]}}

example_user: 
    Query: Is the plaintiff a debt buyer?
    Sources:
    Source 1: 4. Requirements for the complaint.\n\na. The complaint alleges ALL of the following (Civ. Code, §§ 1788.58, 1788.60): (1) That the plaintiff is a debt buyer;\n\n(2) A short, plain statement regarding the nature of the underlying debt and the consumer transaction from which it is derived;
    Source 2: Big Bank is the charge-off creditor. Big Bank sold the debt to the plaintiff.
example_assistant: {{"source_quote_ls": ["Big Bank sold the debt to the plaintiff."]}}

example_user: 
    Query: What is the chain of ownership of the debt?
    Sources:
    Source 1: The debt changed hands several times. The original creditor was Big Bank, which charged off the debt. The debt was then sold to Small Collections Agency, which later sold it to the plaintiff, National Funds LLC.
    Source 2: <table>
    <tr>
    <th>Name</th>
    <th>Contact</th>
    </tr>
    <tr>
    <td>Big Bank</td>
    <td>123 Main Street, San Diego CA 93406</td>
    </tr>
    <tr>
    <td>Small Collections Agency</td>
    <td>45 Circle Avenue, Albany NY 10568</td>
    </tr>
    <tr>
    <td>National Funds LLC</td>
    <td>246 Center Road, Trenton NJ 40958</td>
    </tr>
    </table>
example_assistant: {{"source_quote_ls": ["The original creditor was Big Bank, which charged off the debt. The debt was then sold to Small Collections Agency, which later sold it to the plaintiff, National Funds LLC.", "<table>
    <tr>
    <th>Name</th>
    <th>Contact</th>
    </tr>
    <tr>
    <td>Big Bank</td>
    <td>123 Main Street, San Diego CA 93406</td>
    </tr>
    <tr>
    <td>Small Collections Agency</td>
    <td>45 Circle Avenue, Albany NY 10568</td>
    </tr>
    <tr>
    <td>National Funds LLC</td>
    <td>246 Center Road, Trenton NJ 40958</td>
    </tr>
    </table>"]}}
"""

    source_quote_ls: List[str] = Field(description="The list of quotes that answer the query and are directly quoted from the sources.")

class CitedOutputGeneration(BaseModel):
    """Generate a response to the query grounded in and cited with the source quotes."""

    system_prompt: ClassVar[str] = """Given the query and numbered source quotes, answer the query as succinctly as possible if the source quotes contain an answer. At the end of each sentence of the answer, cite the source quotes used by including the source quote number(s) in square brackets. If the source quotes do not contain an answer, then return \"No answer.\".

Here are some examples:

example_user: 
    Query: What is the debtor's name?
    Source quotes:
    Quote #1: The debtor's name is John Doe.
    Quote #2: The debtor's address is 123 Main St.
example_assistant: {{"cited_output_text": "The debtor's name is John Doe and their address is 123 Main St [1][2]".}}

example_user: 
    Query: What is the charge-off balance in the original complaint?
    Source quotes:
    Quote #1: The charge-off balance in the original complaint is $1,000.
    Quote #2: The charge-off balance in the amended complaint is $1,200.
example_assistant: {{"cited_output_text": "The charge-off balance in the original complaint is $1,000 [1]. The amended complaint has a charge-off balance of $1,200 [2]."}}

example_user:
    Query: Is the plaintiff a debt buyer?
    Source quotes:
    Quote #1: The charge-off creditor sold the debt to the plaintiff.
example_assistant: {{"cited_output_text": "The plaintiff is a debt buyer, as they bought the debt from the charge-off creditor [1]."}}
"""
    cited_output_text: str = Field(description="The answer to the query grounded in the appropriate source quotes with corresponding citation markers at the end of each sentence.")