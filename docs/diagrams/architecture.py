"""README 아키텍처 그림을 만든다.

  pip install diagrams   (graphviz 필요)
  python docs/diagrams/architecture.py

결과: docs/images/architecture.png
실제로 쓰는 구성만 그린다. 클라우드 배포는 없고 전부 로컬에서 python app.py 로 돈다.
"""
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.programming.flowchart import Decision, Document, MultipleDocuments, StoredData
from diagrams.programming.language import Python

FONT = "Apple SD Gothic Neo"
OUT = Path(__file__).resolve().parent.parent / "images" / "architecture"

graph_attr = {"fontname": FONT, "fontsize": "15", "pad": "0.4", "nodesep": "0.55",
              "ranksep": "0.9", "bgcolor": "white", "splines": "spline"}
node_attr = {"fontname": FONT, "fontsize": "12"}
edge_attr = {"fontname": FONT, "fontsize": "11"}

with Diagram("", filename=str(OUT), show=False, direction="LR",
             graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr):
    user = User("python app.py final")

    with Cluster("문서 풀 준비 (한 번만)"):
        arxiv = MultipleDocuments("arXiv 논문 2편\n65쪽")
        ingest = Python("rag/ingest.py\n2단 조판 감지 PyMuPDF")
        index = StoredData("index/\nbge-m3 벡터 + BM25\n청크 211개")

    with Cluster("외부 API"):
        openai = Server("OpenAI\ngpt-4.1-mini 생성\ngpt-4.1 종합/보고서")
        tavily = Server("Tavily\n웹 검색")
        cache = StoredData("web_cache.json\nTTL 14일")

    with Cluster("LangGraph 메인 그래프 (graph.py)"):
        survey = Python("기술 조사\nMLA, ITME 병렬")
        market = Python("시장 평가")
        domain = Python("도메인 평가")
        audit = Python("감사\n찬반, 근거량 계수")
        synth = Python("평가 종합")
        report = Document("보고서 생성")
        verify = Decision("verify()\n코드 검증기")

        with Cluster("Agentic RAG 서브그래프 (세 에이전트 공유)"):
            sub = Python("localize > retrieve\n> grade > widen\n> rewrite > web")

    with Cluster("제출 전 점검 (LLM 없음)"):
        repair = Python("tools/repair.py\n기계적 정정")
        qa = Python("tools/qa.py 12항목\ntools/rubric.py 40항목")
        pdf = Document("Chrome headless\nPDF 8쪽")

    ci = GithubActions("CI\npytest 59개\n비밀키 검사")

    user >> Edge(label="fetch, index") >> arxiv >> ingest >> index
    user >> Edge(label="run --seed 0/1") >> survey
    survey >> Edge(label="Send fan-out") >> [market, domain]
    [market, domain] >> audit
    audit >> Edge(label="기울면 1회 재수집", style="dashed", color="firebrick") >> market
    audit >> Edge(label="통과") >> synth >> report >> verify
    verify >> Edge(label="지적 있으면 재작성 (최대 2회)", style="dashed", color="firebrick") >> report

    [survey, market, domain] >> Edge(style="dotted", color="steelblue") >> sub
    sub >> Edge(label="하이브리드 검색 (RRF)") >> index
    sub >> Edge(label="웹 질의") >> cache >> Edge(label="캐시 미스") >> tavily
    sub >> Edge(label="질의 번역, 관련성 판정", style="dotted") >> openai

    verify >> Edge(label="report_seed*.md, state_seed*.json") >> repair >> qa >> pdf
    ci >> Edge(label="push, PR", style="dotted") >> qa
