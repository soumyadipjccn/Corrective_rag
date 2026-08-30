"""
Prompt templates for Corrective RAG pipeline components.
"""

from langchain_core.prompts import PromptTemplate


GRADE_DOCUMENTS_PROMPT = PromptTemplate(
    template="""You are an expert evaluator grading the relevance of a retrieved document to a user question.
Return ONLY a valid JSON object with a "score" field that is either "yes" or "no".
Do not include markdown backticks, explanations, or any other text.

Document:
{context}

Question:
{question}

Grading Criteria:
- Score "yes" if the document contains keywords, semantic meaning, or factual context related to answering the question.
- Score "no" only if the document is completely unrelated or off-topic.
- Example valid outputs:
  {{"score": "yes"}}
  {{"score": "no"}}""",
    input_variables=["context", "question"],
)


TRANSFORM_QUERY_PROMPT = PromptTemplate(
    template="""You are an AI assistant specialized in web search query optimization.
Generate a refined, search-optimized search engine query that captures the core semantic intent of the user's question.

Original Question:
{question}

Rules:
- Return ONLY the improved query string.
- Do not include quotation marks, explanations, conversational filler, or formatting.
Improved Query:""",
    input_variables=["question"],
)


GENERATE_PROMPT = PromptTemplate(
    template="""You are an accurate, helpful assistant answering questions strictly based on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Provide a clear, comprehensive, and well-structured answer using the facts present in the context.
- If the context does not contain enough information to answer the question, state clearly what is known and what cannot be determined.
- Maintain an objective, professional tone.

Answer:""",
    input_variables=["context", "question"],
)
