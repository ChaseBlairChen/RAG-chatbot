{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c2d5a2df-2a32-4640-a72a-4a5b68cdd257",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting langchain-openai\n",
      "  Downloading langchain_openai-0.3.28-py3-none-any.whl.metadata (2.3 kB)\n",
      "Requirement already satisfied: langchain-core<1.0.0,>=0.3.68 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-openai) (0.3.69)\n",
      "Requirement already satisfied: openai<2.0.0,>=1.86.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-openai) (1.90.0)\n",
      "Collecting tiktoken<1,>=0.7 (from langchain-openai)\n",
      "  Downloading tiktoken-0.9.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.7 kB)\n",
      "Requirement already satisfied: langsmith>=0.3.45 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (0.4.8)\n",
      "Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (9.1.2)\n",
      "Requirement already satisfied: jsonpatch<2.0,>=1.33 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (1.33)\n",
      "Requirement already satisfied: PyYAML>=5.3 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (6.0.2)\n",
      "Requirement already satisfied: typing-extensions>=4.7 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (4.13.2)\n",
      "Requirement already satisfied: packaging>=23.2 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (24.2)\n",
      "Requirement already satisfied: pydantic>=2.7.4 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langchain-core<1.0.0,>=0.3.68->langchain-openai) (2.11.5)\n",
      "Requirement already satisfied: jsonpointer>=1.9 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<1.0.0,>=0.3.68->langchain-openai) (3.0.0)\n",
      "Requirement already satisfied: anyio<5,>=3.5.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (4.9.0)\n",
      "Requirement already satisfied: distro<2,>=1.7.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (1.9.0)\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (0.28.1)\n",
      "Requirement already satisfied: jiter<1,>=0.4.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (0.10.0)\n",
      "Requirement already satisfied: sniffio in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (1.3.1)\n",
      "Requirement already satisfied: tqdm>4 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from openai<2.0.0,>=1.86.0->langchain-openai) (4.67.1)\n",
      "Requirement already satisfied: idna>=2.8 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from anyio<5,>=3.5.0->openai<2.0.0,>=1.86.0->langchain-openai) (3.10)\n",
      "Requirement already satisfied: certifi in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from httpx<1,>=0.23.0->openai<2.0.0,>=1.86.0->langchain-openai) (2025.4.26)\n",
      "Requirement already satisfied: httpcore==1.* in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from httpx<1,>=0.23.0->openai<2.0.0,>=1.86.0->langchain-openai) (1.0.9)\n",
      "Requirement already satisfied: h11>=0.16 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai<2.0.0,>=1.86.0->langchain-openai) (0.16.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic>=2.7.4->langchain-core<1.0.0,>=0.3.68->langchain-openai) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.33.2 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic>=2.7.4->langchain-core<1.0.0,>=0.3.68->langchain-openai) (2.33.2)\n",
      "Requirement already satisfied: typing-inspection>=0.4.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic>=2.7.4->langchain-core<1.0.0,>=0.3.68->langchain-openai) (0.4.1)\n",
      "Requirement already satisfied: regex>=2022.1.18 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from tiktoken<1,>=0.7->langchain-openai) (2024.11.6)\n",
      "Requirement already satisfied: requests>=2.26.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from tiktoken<1,>=0.7->langchain-openai) (2.32.3)\n",
      "Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langsmith>=0.3.45->langchain-core<1.0.0,>=0.3.68->langchain-openai) (3.11.0)\n",
      "Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langsmith>=0.3.45->langchain-core<1.0.0,>=0.3.68->langchain-openai) (1.0.0)\n",
      "Requirement already satisfied: zstandard<0.24.0,>=0.23.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from langsmith>=0.3.45->langchain-core<1.0.0,>=0.3.68->langchain-openai) (0.23.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from requests>=2.26.0->tiktoken<1,>=0.7->langchain-openai) (3.4.2)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from requests>=2.26.0->tiktoken<1,>=0.7->langchain-openai) (2.4.0)\n",
      "Downloading langchain_openai-0.3.28-py3-none-any.whl (70 kB)\n",
      "Downloading tiktoken-0.9.0-cp311-cp311-macosx_11_0_arm64.whl (1.0 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.0/1.0 MB\u001b[0m \u001b[31m19.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hInstalling collected packages: tiktoken, langchain-openai\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2/2\u001b[0m [langchain-openai]\n",
      "\u001b[1A\u001b[2KSuccessfully installed langchain-openai-0.3.28 tiktoken-0.9.0\n"
     ]
    }
   ],
   "source": [
    "!pip install langchain-openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "544976bd-c0fd-42d9-9595-c2a45b2e4704",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fastapi\n",
      "  Downloading fastapi-0.116.1-py3-none-any.whl.metadata (28 kB)\n",
      "Collecting starlette<0.48.0,>=0.40.0 (from fastapi)\n",
      "  Downloading starlette-0.47.2-py3-none-any.whl.metadata (6.2 kB)\n",
      "Requirement already satisfied: pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from fastapi) (2.11.5)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from fastapi) (4.13.2)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.33.2 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (2.33.2)\n",
      "Requirement already satisfied: typing-inspection>=0.4.0 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (0.4.1)\n",
      "Requirement already satisfied: anyio<5,>=3.6.2 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from starlette<0.48.0,>=0.40.0->fastapi) (4.9.0)\n",
      "Requirement already satisfied: idna>=2.8 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from anyio<5,>=3.6.2->starlette<0.48.0,>=0.40.0->fastapi) (3.10)\n",
      "Requirement already satisfied: sniffio>=1.1 in /Users/channiechen/.pyenv/versions/3.11.8/envs/tf-env/lib/python3.11/site-packages (from anyio<5,>=3.6.2->starlette<0.48.0,>=0.40.0->fastapi) (1.3.1)\n",
      "Downloading fastapi-0.116.1-py3-none-any.whl (95 kB)\n",
      "Downloading starlette-0.47.2-py3-none-any.whl (72 kB)\n",
      "Installing collected packages: starlette, fastapi\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2/2\u001b[0m [fastapi]\n",
      "\u001b[1A\u001b[2KSuccessfully installed fastapi-0.116.1 starlette-0.47.2\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "!pip install fastapi\n",
    "from fastapi import FastAPI, Request\n",
    "from pydantic import BaseModel\n",
    "import os\n",
    "from langchain_community.vectorstores import Chroma\n",
    "from langchain.embeddings import HuggingFaceEmbeddings\n",
    "from langchain.prompts import ChatPromptTemplate\n",
    "from langchain_openai import ChatOpenAI\n",
    "\n",
    "CHROMA_PATH = os.path.join(os.path.expanduser(\"~\"), \"my_chroma_db\")\n",
    "\n",
    "PROMPT_TEMPLATE = \"\"\"\n",
    "Answer the question based only on the following factual context and use a little bit of your understanding as well:\n",
    "{context}\n",
    "\n",
    "Question: {question}\n",
    "Answer using as many words as possible with a serious tone like a lawyer:\n",
    "\"\"\"\n",
    "\n",
    "class Query(BaseModel):\n",
    "    question: str\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "@app.post(\"/ask\")\n",
    "def ask_question(query: Query):\n",
    "    query_text = query.question\n",
    "\n",
    "    embedding_function = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")\n",
    "    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)\n",
    "    results = db.similarity_search_with_relevance_scores(query_text, k=3)\n",
    "    \n",
    "    if not results or results[0][1] < 0.5:\n",
    "        return {\"response\": \"Nothing in the records.\"}\n",
    "\n",
    "    context_text = \"\\n\\n---\\n\\n\".join([doc.page_content for doc, _ in results])\n",
    "    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)\n",
    "\n",
    "    llm = ChatOpenAI(\n",
    "        model=\"deepseek/deepseek-chat\",\n",
    "        temperature=0.2,\n",
    "        max_tokens=200,\n",
    "        openai_api_key=os.environ[\"OPENAI_API_KEY\"],\n",
    "        openai_api_base=os.environ[\"OPENAI_API_BASE\"]\n",
    "    )\n",
    "    response = llm.predict(prompt)\n",
    "    return {\"response\": response}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8523f8e0-9d08-4523-8bbe-bdaf399b8187",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4360b0c7-b765-4b8e-a87d-3d034a7ab3af",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
