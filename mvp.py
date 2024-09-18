import os
import json
import random # 목업 스코어용
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.tools import Tool
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Gemma 모델 로드 함수
def load_gemma_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

# HR 비서 에이전트 프롬프트
hr_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 스타트업의 HR 비서입니다. 스타트업에 대한 정보를 수집하여 적합한 인재를 찾는 것이 목표입니다. 질문은 간결하고 명확해야 합니다."),
    ("human", "{input}"),
    ("ai", "{output}"),
    ("human", "다음 질문을 해주세요:")
])

# 스타트업 에이전트 프롬프트
startup_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 스타트업의 채용담당자입니다. HR 비서의 질문에 답변해주세요."),
    ("human", "{input}"),
    ("ai", "{output}"),
    ("human", "HR 비서의 질문: {question}")
])

# 평가자 에이전트 프롬프트
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 HR 비서와 스타트업 채용담당자 간의 대화를 평가하는 평가자입니다. 각 질문-답변 쌍에 대해 1-10점 척도로 점수를 매겨주세요."),
    ("human", "다음 대화를 평가해주세요:\n\nHR 비서: {question}\n\n스타트업 채용담당자: {answer}")
])

# 인재 검색 및 평가 도구
@tool
def search_and_evaluate_candidates(summary: str) -> Dict[str, Any]:
    """주어진 요약을 바탕으로 인재를 검색하고 평가합니다."""
    # 실제 구현에서는 데이터베이스 검색 및 match_score 평가 로직이 들어갑니다
    candidates = [
        {"name": "후보자1", "skills": ["Python", "ML"], "match_score": 0.8},
        {"name": "후보자2", "skills": ["Java", "Web"], "match_score": 0.7},
        # ... 더 많은 후보자
    ]
    avg_match_score = sum(c["match_score"] for c in candidates) / len(candidates)
    return {
        "candidates": candidates,
        "avg_match_score": avg_match_score,
        "satisfaction_score": avg_match_score * random.random() # 실제로는 well-designed 평가 함수 사용
    }

# 도구 실행기 설정
tools = [Tool.from_function(search_and_evaluate_candidates)]
tool_executor = ToolExecutor(tools)

# 그래프 상태 정의
class AgentState(Dict):
    """대화 상태를 나타내는 클래스"""
    conversation: List[Dict[str, Any]]
    summary: str = ""
    hr_score: float = 0.0
    enough_info: bool = False

# 대화 기록 저장 함수 (수정됨)
def save_conversation(conversation: List[Dict[str, Any]], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

# 컨텍스트 생성 함수
def generate_context(conversation: List[Dict[str, Any]]) -> str:
    context = ""
    for entry in conversation:
        if 'question' in entry and 'answer' in entry:
            context += f"Q: {entry['question']}\nA: {entry['answer']}\n\n"
    return context.strip()

# HR 비서 노드
def hr_agent_node(state: AgentState) -> Dict[str, Any]:
    if state["enough_info"]:
        return {"next_node": "search_node"}
    
    context = generate_context(state["conversation"])
    hr_message = hr_prompt.invoke({"input": context, "output": ""})
    question = hr_agent(hr_message.to_string())
    
    if "<ENOUGH INFORMATION>" in question:
        state["enough_info"] = True
        state["summary"] = hr_agent(f"Summarize the following conversation:\n{context}")
        return {"next_node": "search_node"}
    
    new_entry = {
        "context": context,
        "question": question
    }
    state["conversation"].append(new_entry)
    save_conversation(state["conversation"], "conversation_log.json")
    return {"next_node": "startup_node"}

# 스타트업 노드
def startup_agent_node(state: AgentState) -> Dict[str, Any]:
    last_entry = state["conversation"][-1]
    startup_message = startup_prompt.invoke({"input": last_entry["context"], "output": "", "question": last_entry["question"]})
    answer = startup_agent.invoke(startup_message.to_messages()).content
    last_entry["answer"] = answer
    save_conversation(state["conversation"], "conversation_log.json")
    return {"next_node": "evaluator_node"}

# 평가자 노드
def evaluator_agent_node(state: AgentState) -> Dict[str, Any]:
    last_entry = state["conversation"][-1]
    evaluator_message = evaluator_prompt.invoke({
        "question": last_entry["question"],
        "answer": last_entry["answer"]
    })
    evaluation = evaluator_agent.invoke(evaluator_message.to_messages()).content
    score = int(evaluation.split()[0])  # 첫 번째 단어를 점수로 가정
    last_entry["score"] = score
    save_conversation(state["conversation"], "conversation_log.json")
    return {"next_node": "hr_node"}

# 검색 노드
def search_node(state: AgentState) -> Dict[str, Any]:
    result = tool_executor.invoke({"name": "search_and_evaluate_candidates", "arguments": {"summary": state["summary"]}})
    state["search_result"] = result
    state["hr_score"] = sum(entry["score"] for entry in state["conversation"] if "score" in entry) / len(state["conversation"])
    state["conversation"].append({"search_result": result})
    save_conversation(state["conversation"], "conversation_log.json")
    return {"next_node": END}

# 그래프 정의
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("hr_node", hr_agent_node)
workflow.add_node("startup_node", startup_agent_node)
workflow.add_node("evaluator_node", evaluator_agent_node)
workflow.add_node("search_node", search_node)

# 엣지 연결
workflow.add_edge("hr_node", "startup_node")
workflow.add_edge("startup_node", "evaluator_node")
workflow.add_edge("evaluator_node", "hr_node")
workflow.add_edge("hr_node", "search_node")
workflow.set_entry_point("hr_node")

# 그래프 컴파일
app = workflow.compile()

# 실행
def run_workflow():
    inputs = AgentState(conversation=[])
    for output in app.stream(inputs):
        if output["next_node"] == END:
            final_state = output
            break

    print(f"Final HR Score: {final_state['hr_score']}")
    print(f"Candidates: {final_state['search_result']['candidates']}")
    print(f"Average Match Score: {final_state['search_result']['avg_match_score']}")
    print(f"Satisfaction Score: {final_state['search_result']['satisfaction_score']}")

if __name__ == "__main__":
    # HR 비서 에이전트 (Gemma 모델)
    hr_agent = load_gemma_model("path/to/gemma/model")

    # 다른 에이전트 정의
    startup_agent = ChatOpenAI(model="gpt-4", temperature=0.5)
    evaluator_agent = ChatOpenAI(model="gpt-4", temperature=0.2)
    
    run_workflow()