"""
build_flowise_template.py — generates the Flowise chatflow template
`query_rerank_redismem.json` (multi-query + Cohere rerank + Redis memory) and a fallback
`query_rerank_redismem_basic.json` (no multi-query) for Flowise versions whose canvas
refuses the Multi-Query → Cohere-Rerank link.

    OpenAI Embeddings ─► Chroma (collection company_kb, http://localhost:8000, filter tenant=northwind)
                            │ vector store
                            ▼
    ChatOpenAI ───────► Multi Query Retriever  (LLM writes 3 query variants, union of results)
        │                   │ retriever
        │                   ▼
        │            Cohere Rerank Retriever    (top 5)
        │                   │ retriever
        ▼                   ▼
    Conversational Retrieval QA Chain  ◄── Redis-Backed Chat Memory (window 6, TTL 1 day)

Run:  python flowise/build_flowise_template.py
Then: Flowise → Chatflows → Add New → ⚙ (top-right) → Load Chatflow → pick the JSON.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

RESPONSE_PROMPT = (
    "You are the internal Company Knowledge-Base assistant. Answer ONLY from the context below. "
    "After every factual sentence cite the source document title in square brackets, e.g. "
    "[Leave Policy v4.0 (2026)]. If passages conflict, prefer the most recent effective date. "
    "Quote IDs, amounts and dates exactly. If the context does not contain the answer, reply exactly: "
    "\"I couldn't find this in the knowledge base.\"\n------------\n{context}\n------------"
)
REPHRASE_PROMPT = (
    "Given the following conversation and a follow up question, rephrase the follow up question to be "
    "a standalone question. Keep any IDs exactly.\n\nChat History:\n{chat_history}\n"
    "Follow Up Input: {question}\nStandalone Question:"
)
MULTI_QUERY_PROMPT = (
    "You are an AI assistant searching a company knowledge base (policies, runbooks, contracts, tickets). "
    "Generate 3 different versions of the user question using synonyms and the formal terms a policy "
    "document would use, keeping any IDs exactly. Provide them separated by newlines.\n"
    "Original question: {question}"
)


def param(label, name, type_, node_id, optional=False, **extra):
    return {"label": label, "name": name, "type": type_, "optional": optional, "id": f"{node_id}-input-{name}-{type_}", **extra}


def anchor(label, name, type_, node_id, optional=False, list_=False):
    a = {"label": label, "name": name, "type": type_, "id": f"{node_id}-input-{name}-{type_}"}
    if optional:
        a["optional"] = True
    if list_:
        a["list"] = True
    return a


def out_anchor(node_id, name, label, classes):
    return {"id": f"{node_id}-output-{name}-{'|'.join(classes)}", "name": name, "label": label, "type": " | ".join(classes)}


def node(node_id, x, y, *, label, name, type_, version, category, description, base_classes,
         input_params, input_anchors, inputs, output_anchors, outputs=None, height=500):
    return {
        "id": node_id, "position": {"x": x, "y": y}, "positionAbsolute": {"x": x, "y": y},
        "type": "customNode", "width": 300, "height": height, "selected": False, "dragging": False,
        "data": {
            "id": node_id, "label": label, "name": name, "type": type_, "version": version,
            "category": category, "description": description, "baseClasses": base_classes,
            "inputParams": input_params, "inputAnchors": input_anchors, "inputs": inputs,
            "outputAnchors": output_anchors, "outputs": outputs or {}, "selected": False,
        },
    }


def edge(src, src_handle, tgt, tgt_handle):
    return {"source": src, "sourceHandle": src_handle, "target": tgt, "targetHandle": tgt_handle,
            "type": "buttonedge", "id": f"{src}-{src_handle}-{tgt}-{tgt_handle}"}


def build(with_multi_query: bool) -> dict:
    nodes, edges = [], []

    # --- ChatOpenAI
    cid = "chatOpenAI_0"
    chat_cls = ["ChatOpenAI", "BaseChatModel", "BaseLanguageModel", "Runnable"]
    nodes.append(node(cid, 80, 60, label="ChatOpenAI", name="chatOpenAI", type_="ChatOpenAI", version=8,
                      category="Chat Models", description="OpenAI chat model", base_classes=chat_cls,
                      input_params=[
                          param("Connect Credential", "credential", "credential", cid, credentialNames=["openAIApi"]),
                          param("Model Name", "modelName", "asyncOptions", cid, loadMethod="listModels", default="gpt-4o-mini"),
                          param("Temperature", "temperature", "number", cid, optional=True, step=0.1, default=0.9),
                          param("Streaming", "streaming", "boolean", cid, optional=True, default=True),
                      ],
                      input_anchors=[anchor("Cache", "cache", "BaseCache", cid, optional=True)],
                      inputs={"cache": "", "modelName": "gpt-4o-mini", "temperature": 0, "streaming": True},
                      output_anchors=[out_anchor(cid, "chatOpenAI", "ChatOpenAI", chat_cls)], height=670))

    # --- OpenAI embeddings
    eid = "openAIEmbeddings_0"
    emb_cls = ["OpenAIEmbeddings", "Embeddings"]
    nodes.append(node(eid, 80, 780, label="OpenAI Embeddings", name="openAIEmbeddings", type_="OpenAIEmbeddings",
                      version=4, category="Embeddings", description="Must match the model used by m2_chroma_ingestion.py",
                      base_classes=emb_cls,
                      input_params=[
                          param("Connect Credential", "credential", "credential", eid, credentialNames=["openAIApi"]),
                          param("Model Name", "modelName", "asyncOptions", eid, loadMethod="listModels", default="text-embedding-3-small"),
                      ],
                      input_anchors=[], inputs={"modelName": "text-embedding-3-small"},
                      output_anchors=[out_anchor(eid, "openAIEmbeddings", "OpenAIEmbeddings", emb_cls)], height=420))

    # --- Chroma (reads the collection our Python pipeline built)
    vid = "chroma_0"
    vs_out = "vectorStore" if with_multi_query else "retriever"
    nodes.append(node(vid, 460, 700, label="Chroma", name="chroma", type_="Chroma", version=2,
                      category="Vector Stores", description="Existing collection 'company_kb' served by `chroma run`",
                      base_classes=["Chroma", "VectorStoreRetriever", "BaseRetriever"],
                      input_params=[
                          param("Connect Credential", "credential", "credential", vid, optional=True, credentialNames=["chromaApi"]),
                          param("Collection Name", "collectionName", "string", vid),
                          param("Chroma URL", "chromaURL", "string", vid, optional=True),
                          param("Chroma Metadata Filter", "chromaMetadataFilter", "json", vid, optional=True, additionalParams=True),
                          param("Top K", "topK", "number", vid, optional=True, additionalParams=True, step=1, default=4),
                      ],
                      input_anchors=[
                          anchor("Document", "document", "Document", vid, optional=True, list_=True),
                          anchor("Embeddings", "embeddings", "Embeddings", vid),
                          anchor("Record Manager", "recordManager", "RecordManager", vid, optional=True),
                      ],
                      inputs={"document": "", "embeddings": f"{{{{{eid}.data.instance}}}}", "recordManager": "",
                              "collectionName": "company_kb", "chromaURL": "http://localhost:8000",
                              "chromaMetadataFilter": json.dumps({"tenant": "northwind"}), "topK": 50},
                      output_anchors=[{"name": "output", "label": "Output", "type": "options", "default": "retriever",
                                       "options": [
                                           out_anchor(vid, "retriever", "Chroma Retriever", ["Chroma", "VectorStoreRetriever", "BaseRetriever"]),
                                           out_anchor(vid, "vectorStore", "Chroma Vector Store", ["Chroma", "VectorStore"]),
                                       ]}],
                      outputs={"output": vs_out}, height=600))
    edges.append(edge(eid, f"{eid}-output-openAIEmbeddings-OpenAIEmbeddings|Embeddings", vid, f"{vid}-input-embeddings-Embeddings"))

    rerank_input_src = None
    # --- Multi Query Retriever
    if with_multi_query:
        mid = "multiQueryRetriever_0"
        mq_cls = ["MultiQueryRetriever", "BaseRetriever", "Runnable"]
        nodes.append(node(mid, 860, 420, label="Multi Query Retriever", name="multiQueryRetriever",
                          type_="MultiQueryRetriever", version=1, category="Retrievers",
                          description="LLM generates 3 query variants; union of their results",
                          base_classes=mq_cls,
                          input_params=[param("Prompt", "modelPrompt", "string", mid, rows=4, default=MULTI_QUERY_PROMPT)],
                          input_anchors=[anchor("Vector Store", "vectorStore", "VectorStore", mid),
                                         anchor("Language Model", "model", "BaseLanguageModel", mid)],
                          inputs={"vectorStore": f"{{{{{vid}.data.instance}}}}", "model": f"{{{{{cid}.data.instance}}}}",
                                  "modelPrompt": MULTI_QUERY_PROMPT},
                          output_anchors=[out_anchor(mid, "multiQueryRetriever", "MultiQueryRetriever", mq_cls)], height=500))
        edges.append(edge(vid, f"{vid}-output-vectorStore-Chroma|VectorStore", mid, f"{mid}-input-vectorStore-VectorStore"))
        edges.append(edge(cid, f"{cid}-output-chatOpenAI-{'|'.join(chat_cls)}", mid, f"{mid}-input-model-BaseLanguageModel"))
        rerank_input_src = (mid, f"{mid}-output-multiQueryRetriever-{'|'.join(mq_cls)}")
    else:
        rerank_input_src = (vid, f"{vid}-output-retriever-Chroma|VectorStoreRetriever|BaseRetriever")

    # --- Cohere Rerank Retriever (stage 2: top-50 → top-5)
    rid = "cohereRerankRetriever_0"
    rr_cls = ["Cohere Rerank Retriever", "BaseRetriever"]
    nodes.append(node(rid, 1240, 520, label="Cohere Rerank Retriever", name="cohereRerankRetriever",
                      type_="Cohere Rerank Retriever", version=1, category="Retrievers",
                      description="Cross-encoder rerank of the candidate pool", base_classes=rr_cls,
                      input_params=[
                          param("Connect Credential", "credential", "credential", rid, credentialNames=["cohereApi"]),
                          param("Model Name", "model", "options", rid, default="rerank-english-v3.0", optional=True,
                                options=[{"label": "rerank-english-v3.0", "name": "rerank-english-v3.0"},
                                         {"label": "rerank-multilingual-v3.0", "name": "rerank-multilingual-v3.0"}]),
                          param("Query", "query", "string", rid, optional=True, acceptVariable=True),
                          param("Top K", "topK", "number", rid, optional=True, additionalParams=True, default=4),
                          param("Max Chunks Per Doc", "maxChunksPerDoc", "number", rid, optional=True, additionalParams=True, default=10),
                      ],
                      input_anchors=[anchor("Vector Store Retriever", "baseRetriever", "VectorStoreRetriever", rid)],
                      inputs={"baseRetriever": f"{{{{{rerank_input_src[0]}.data.instance}}}}", "model": "rerank-english-v3.0",
                              "query": "", "topK": 5, "maxChunksPerDoc": 10},
                      output_anchors=[{"name": "output", "label": "Output", "type": "options", "default": "retriever",
                                       "options": [
                                           out_anchor(rid, "retriever", "Cohere Rerank Retriever", rr_cls),
                                           out_anchor(rid, "document", "Document", ["Document", "json"]),
                                           out_anchor(rid, "text", "Text", ["string", "json"]),
                                       ]}],
                      outputs={"output": "retriever"}, height=560))
    edges.append(edge(rerank_input_src[0], rerank_input_src[1], rid, f"{rid}-input-baseRetriever-VectorStoreRetriever"))

    # --- Redis-backed memory
    mid2 = "RedisBackedChatMemory_0"
    mem_cls = ["RedisBackedChatMemory", "BaseChatMemory", "BaseMemory"]
    nodes.append(node(mid2, 1240, 40, label="Redis-Backed Chat Memory", name="RedisBackedChatMemory",
                      type_="RedisBackedChatMemory", version=2, category="Memory",
                      description="Conversation memory persisted in Redis (survives restarts, shared across replicas)",
                      base_classes=mem_cls,
                      input_params=[
                          param("Connect Credential", "credential", "credential", mid2, credentialNames=["redisCacheApi", "redisCacheUrlApi"]),
                          param("Session Id", "sessionId", "string", mid2, optional=True, additionalParams=True, default=""),
                          param("Session Timeouts", "sessionTTL", "number", mid2, optional=True, additionalParams=True),
                          param("Memory Key", "memoryKey", "string", mid2, additionalParams=True, default="chat_history"),
                          param("Window Size", "windowSize", "number", mid2, optional=True, additionalParams=True),
                      ],
                      input_anchors=[],
                      inputs={"sessionId": "", "sessionTTL": 86400, "memoryKey": "chat_history", "windowSize": 6},
                      output_anchors=[out_anchor(mid2, "RedisBackedChatMemory", "RedisBackedChatMemory", mem_cls)], height=330))

    # --- Conversational Retrieval QA Chain
    qid = "conversationalRetrievalQAChain_0"
    qa_cls = ["ConversationalRetrievalQAChain", "BaseChain", "Runnable"]
    nodes.append(node(qid, 1640, 300, label="Conversational Retrieval QA Chain", name="conversationalRetrievalQAChain",
                      type_="ConversationalRetrievalQAChain", version=3, category="Chains",
                      description="Rephrase with memory → retrieve → answer with citations", base_classes=qa_cls,
                      input_params=[
                          param("Return Source Documents", "returnSourceDocuments", "boolean", qid, optional=True),
                          param("Rephrase Prompt", "rephrasePrompt", "string", qid, optional=True, additionalParams=True, rows=4, default=REPHRASE_PROMPT),
                          param("Response Prompt", "responsePrompt", "string", qid, optional=True, additionalParams=True, rows=4, default=RESPONSE_PROMPT),
                      ],
                      input_anchors=[
                          anchor("Chat Model", "model", "BaseChatModel", qid),
                          anchor("Vector Store Retriever", "vectorStoreRetriever", "BaseRetriever", qid),
                          anchor("Memory", "memory", "BaseMemory", qid, optional=True),
                          anchor("Input Moderation", "inputModeration", "Moderation", qid, optional=True, list_=True),
                      ],
                      inputs={"model": f"{{{{{cid}.data.instance}}}}", "vectorStoreRetriever": f"{{{{{rid}.data.instance}}}}",
                              "memory": f"{{{{{mid2}.data.instance}}}}", "returnSourceDocuments": True,
                              "rephrasePrompt": REPHRASE_PROMPT, "responsePrompt": RESPONSE_PROMPT, "inputModeration": ""},
                      output_anchors=[out_anchor(qid, "conversationalRetrievalQAChain", "ConversationalRetrievalQAChain", qa_cls)],
                      height=530))
    edges.append(edge(cid, f"{cid}-output-chatOpenAI-{'|'.join(chat_cls)}", qid, f"{qid}-input-model-BaseChatModel"))
    edges.append(edge(rid, f"{rid}-output-retriever-{'|'.join(rr_cls)}", qid, f"{qid}-input-vectorStoreRetriever-BaseRetriever"))
    edges.append(edge(mid2, f"{mid2}-output-RedisBackedChatMemory-{'|'.join(mem_cls)}", qid, f"{qid}-input-memory-BaseMemory"))
    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    for name, mq in (("query_rerank_redismem.json", True), ("query_rerank_redismem_basic.json", False)):
        flow = build(mq)
        (OUT / name).write_text(json.dumps(flow, indent=2))
        print(f"wrote flowise/{name}: {len(flow['nodes'])} nodes, {len(flow['edges'])} edges")
