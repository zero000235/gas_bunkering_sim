import numpy as np
import pandas as pd
import time
import json
import threading
import streamlit as st
from functools import lru_cache
from http.server import HTTPServer, BaseHTTPRequestHandler
try:
    import CoolProp.CoolProp as CP
except ImportError:
    pass

import urllib.parse

# 🛡️ [다중 사용자 세션 격리 아키텍처] 
# 세션별 독립 도면 저장을 위한 격리 맵 선언
SESSION_CAD_DATA = {}
SESSION_LOCK = threading.Lock()
SESSION_UPDATED = {}


class CadBridgeHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 콘솔 로그 지저분화 방지
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/sync"):
            # URL 파라미터에서 session_id 파싱
            parsed_path = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_path.query)
            session_id = query.get("session_id", ["default"])[0]

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Run real-time physical simulation on current drawing elements
                pipes_list = data.get("pipes", [])
                nodes_list = data.get("nodes", [])
                
                if pipes_list and nodes_list:
                    try:
                        # 0.01s rapid physics solving using default fluid properties (Water, Carbon Steel)
                        solve_pipe_network(pipes_list, nodes_list, 998.2, 0.001002, 4.5e-5, 120.0, "Commercial Steel (상업용 강관)")
                        clean_numerical_results(pipes_list, nodes_list)
                    except Exception:
                        pass
                
                with SESSION_LOCK:
                    SESSION_CAD_DATA[session_id] = data
                    SESSION_UPDATED[session_id] = True
                
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                # 반환용 JSON 직렬화 전 2차 수치 세척
                clean_numerical_results(pipes_list, nodes_list)
                self.wfile.write(json.dumps({
                    "status": "success",
                    "pipes": pipes_list,
                    "nodes": nodes_list
                }, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

@st.cache_resource
def start_bridge_server():
    for port in [18501, 18502, 18503]:
        try:
            # 로컬 루프백 127.0.0.1 바인딩
            server = HTTPServer(('127.0.0.1', port), CadBridgeHTTPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return port
        except Exception:
            continue
    return 18501  # 만약 모든 포트 충돌 시, 기존에 떠 있는 18501 포트를 신뢰하여 폴백 활용 보장

try:
    import CoolProp.CoolProp as CP
except ImportError:
    st.error("CoolProp 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해 주세요.")
    st.stop()

# Gemini AI 모듈 가져오기 (오류 방지용 예외 처리)
try:
    import google.generativeai as genai
except ImportError:
    st.warning("google-generativeai 라이브러리가 설치되지 않아 AI 도면 분석 기능이 일부 제한될 수 있습니다.")

# =============================================================================
# [공통 데이터 및 물성치/계산 유틸리티]
# =============================================================================

FLUID_OPTIONS = {
    "Water"            : "물 (Water)",
    "Methanol"         : "메탄올 (Methanol)",
    "Ethanol"          : "에탄올 (Ethanol)",
    "INCOMP::MEG[0.5]" : "에틸렌 글리콜 (부동액 50% 수용액)",
    "INCOMP::MPG[0.5]" : "프로필렌 글리콜 (부동액 50% 수용액)",
    "Acetone"          : "아세톤 (Acetone)",
    "Benzene"          : "벤젠 (Benzene)",
    "Toluene"          : "톨루엔 (Toluene)",
}

FLUID_MATERIAL_RECOMMENDATIONS = {
    "Water": {
        "best": ["Stainless Steel (스테인리스 강관)", "PVC (일반 플라스틱 관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Cast Iron (주철관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Smooth Pipe (초매끈한 관, ε=0)"],
        "reason": "물은 범용적인 유체로 위생성과 내부식성이 우수한 스테인리스강 및 가볍고 녹슬지 않는 PVC가 최적입니다. 탄소강(Commercial Steel)은 장기 기동 시 부식성 스케일이 누적되어 수력 마찰이 증가할 수 있어 아연도금이나 주철관이 차선책이 됩니다.",
        "connection": "용접 체결 (Welded) 또는 플랜지 체결 (Flanged)",
        "support": "일반 스프링 행거 (Spring Hanger) 및 롤러 서포트, 배관 지지 간격 2.5m ~ 3.0m 권장"
    },
    "Methanol": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "메탄올과 같은 저급 알코올류 유기용제는 장기 노출 시 일반 플라스틱(PVC) 수지의 사슬 구조를 팽창시키거나 연화시켜 미세 누출 및 균열을 야기합니다. 화학적 안정성이 뛰어난 스테인리스강이나 강도가 확보된 상업용 강관 사용이 강제됩니다.",
        "connection": "완전 기밀 용접 체결 (Welded - 미세 알코올 누출 차단용)",
        "support": "슬라이딩 슈 (Sliding Shoe) 서포트 및 U-볼트 가이드, 진동 제어용 2.0m 간격 권장"
    },
    "Ethanol": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "에탄올은 친유성 및 침투성이 있어 PVC 수지를 서서히 침식하여 결합부를 파손합니다. 화학적으로 비활성이며 내화학 장벽을 형성하는 스테인리스강이나 강관 계열을 필히 권장합니다.",
        "connection": "용접 체결 (Welded) 권장 (기밀 장벽 밀폐성 확보용)",
        "support": "인슐레이션 보온재 보호형 슬라이딩 슈 및 U-볼트, 열신축 고려 2.0m 간격 권장"
    },
    "INCOMP::MEG[0.5]": {
        "best": ["Stainless Steel (스테인리스 강관)", "Galvanized Steel (아연도금 강관)"],
        "ok": ["Commercial Steel (상업용 강관)", "PVC (일반 플라스틱 관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Cast Iron (주철관)"],
        "reason": "에틸렌글리콜 50% 수용액은 산소와 결합 시 글리콜산 등의 유기산으로 분해되어 일반 주철관의 탈탄소 및 급격한 부식을 유발할 수 있습니다. 이를 방지하기 위해 방청 처리가 우수한 아연도금 강관 또는 스테인리스 강관이 탁월합니다.",
        "connection": "플랜지 체결 (Flanged - 밸브 등 분해 정비 다수 구역) 및 용접 체결",
        "support": "리지드 행거 (Rigid Hanger) 및 진동 감쇄용 스프링 행거, 배관 처짐 방지용 2.5m 간격 권장"
    },
    "INCOMP::MPG[0.5]": {
        "best": ["Stainless Steel (스테인리스 강관)", "PVC (일반 플라스틱 관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Commercial Steel (상업용 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Cast Iron (주철관)"],
        "reason": "프로필렌글리콜은 독성이 적어 식품/제약용 냉매로 다수 쓰이므로, 위생 등급을 만족하는 스테인리스 강관이나 PVC 플라스틱이 베스트 소재입니다. 일반 주철관은 부식 생성물이 냉매 유로를 막을 위험이 큽니다.",
        "connection": "용접 체결 (Welded - 위생 등급 및 무독성 차단) 또는 플랜지 체결",
        "support": "방진 고무 패드 패킹 스프링 행거 및 지지 가이드, 냉동 하중 분산용 2.5m 간격 권장"
    },
    "Acetone": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)", "Smooth Pipe (초매끈한 관, ε=0)"],
        "reason": "아세톤은 극성을 띤 매우 강력한 케톤계 유기용제로, 플라스틱(PVC) 수지를 즉각적으로 부풀리고 흐물흐물하게 녹여버립니다. 플라스틱 배관 설계 시 대형 폭발/누출 사고로 직결되므로, 반드시 강인한 탄소강관이나 내식성이 탁월한 스테인리스 강관을 사용하셔야 합니다.",
        "connection": "완전 기밀 용접 체결 (Welded - 플라스틱 가스켓 용해 차단 필수)",
        "support": "강제 앵커 (Anchor System) 및 리지드 서포트, 극미세 기화 압력 수격 대비 1.8m 초조밀 간격 필수"
    },
    "Benzene": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)", "Cast Iron (주철관)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "벤젠은 방향족 탄화수소로 고분자 폴리머(PVC)를 격렬히 침식하고 가소제를 용출시켜 배관을 경화 및 파열시킵니다. 구조적 강도와 우수한 밀폐성을 제공하는 금속제 스테인리스강 또는 탄소강관을 강력 추천합니다.",
        "connection": "완전 올-웰디드 밀폐 용접 체결 (Welded - 1급 발암 물질 누출 원천 봉쇄)",
        "support": "특수 방진 스프링 서포트 및 고중량 강재 앵커, 센서 연계용 2.0m 간격 견고 지지 권장"
    },
    "Toluene": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)", "Cast Iron (주철관)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "톨루엔은 벤젠계 용제로서 강한 친지성을 가져 PVC 분자 사슬 결합을 깨뜨리고 체적 팽창을 일으켜 이음부 파손을 일으킵니다. 내유성 및 고온/고압 기밀이 확보되는 스테인리스 강관 및 탄소강관이 절실히 요구됩니다.",
        "connection": "완전 올-웰디드 기밀 용접 체결 (Welded - 인화 휘발성 완벽 차단)",
        "support": "헤비 듀티 슬라이딩 서포트 (Heavy-Duty Sliding Support) 및 가이드, 열신축 완화 2.0m 간격 권장"
    }
}

def calculate_dynamic_pump_efficiency(q_m3s: float, dp_pa: float) -> float:
    """
    설계 배관망 상태(유량 및 필요 양정 압력강하)를 받아,
    수역학적 유량별 펌프 단독 효율과 펌프 제어 마력별 모터 효율(IE3급 프리미엄)을 계산하여
    종합 설계 효율(%)을 동적으로 자동 연산합니다.
    """
    q_m3h = abs(q_m3s) * 3600.0
    if q_m3h <= 0.01:
        return 50.0 # 초유동 상태 최소 폴백 효율 50%
        
    # 1. 수량(Flow Rate) 스케일에 따른 펌프 단독 유압 수력 효율 근사 (유량이 클수록 최적 수력 매칭으로 80% 근접)
    eta_pump = 0.78 - 0.38 * np.exp(-0.08 * q_m3h)
    eta_pump = max(0.35, min(0.82, eta_pump))
    
    # 2. 필요 유동 마력에 따른 전동기(IE3 프리미엄 모터) 기준 모터 효율 근사
    p_hydro_w = abs(dp_pa) * abs(q_m3s)
    p_shaft_kw = (p_hydro_w / eta_pump) / 1000.0
    
    if p_shaft_kw <= 0.4:
        eta_motor = 0.72
    elif p_shaft_kw <= 1.5:
        eta_motor = 0.80
    elif p_shaft_kw <= 7.5:
        eta_motor = 0.85
    elif p_shaft_kw <= 37.0:
        eta_motor = 0.90
    else:
        eta_motor = 0.93
        
    total_efficiency_percent = (eta_pump * eta_motor) * 100.0
    return round(max(30.0, min(95.0, total_efficiency_percent)), 1)

def recommend_pipe_spec(q_m3s, material_name, min_d_m=0.0):
    """
    Genereaux 식 기반 LCC 최소화 기법 및 경제 유속 한계를 통합한 최적 상용 KS 관경 산출 엔진
    (NPSH 방지를 위해 격상된 최소 가이딩 내경 min_d_m 동적 일치화 보장)
    """
    if q_m3s <= 0:
        q_m3s = 1e-9
        
    # [1] Genereaux 모델을 기반으로 한 LCC 최소화 최적 내경
    d_opt_gen = 0.37 * (q_m3s ** 0.45)
    
    # [2] 동력 소비량과 초기 시공비의 수력학적 평형 유속 (1.2 m/s) 가이드
    v_opt = 1.2
    d_opt_vel = np.sqrt((4.0 * q_m3s) / (np.pi * v_opt))
    
    # 가중 평균을 통한 최종 설계 목표 내경(d_opt) 확정
    d_opt = 0.65 * d_opt_gen + 0.35 * d_opt_vel
    
    # NPSH 안전 가이드 직경이 존재할 시 강제 격상 조율
    if d_opt < min_d_m:
        d_opt = min_d_m
    
    is_stainless = "Stainless" in material_name
    std_key = "KS D 3576 (스테인리스 강관)" if is_stainless else "KS D 3562 (압력 배관용 탄소강관)"
    std_data = PIPE_STANDARDS[std_key]
    
    best_nps = "15A"
    best_sch = "SCH 10S" if is_stainless else "SCH 40"
    best_id_m = 0.015
    min_diff = float('inf')
    
    for nps, info in std_data["data"].items():
        od = info["OD"] / 1000.0
        schedules = [sch for sch in std_data["schedules"] if sch in info]
        for sch in schedules:
            t = info[sch] / 1000.0
            internal_d = od - 2.0 * t
            if internal_d <= 0:
                continue
                
            # 만약 min_d_m 가이드라인보다 규격 내경이 작다면 안전을 위해 격상 스킵
            if min_d_m > 0 and internal_d < min_d_m:
                continue
                
            # 격상된 설계 내경과의 편차 최소화 매칭
            diff = abs(internal_d - d_opt)
            if diff < min_diff:
                min_diff = diff
                best_nps = nps
                best_sch = sch
                best_id_m = internal_d
                
    nps_clean = best_nps.split(" ")[0]
    return best_id_m, f"{nps_clean} - {best_sch}"

def align_pipe_network_topology(pipes_list, nodes_list):
    """
    유동 위상 자동 정렬 (Flow Topology Alignment) 엔진:
    사용자가 파이프를 그린 드래그 방향과 무관하게, 펌프(Source/출발지)에서 출발하여
    아웃렛/토출구(Junction/Tank/Sink)로 도달하는 물리적 실제 흐름 방향으로 from/to를 내부 자동 Re-orient 정렬함.
    """
    node_map = {n["id"]: n for n in nodes_list}
    pump_nodes = [n["id"] for n in nodes_list if n["type"] == "pump"]
    tank_nodes = [n["id"] for n in nodes_list if n["type"] == "tank" or "OUTLET" in n.get("name", "")]
    
    if not pump_nodes:
        # 펌프가 없을 경우 연결 첫 노드를 임시 소스로 활용
        sources = [n["id"] for n in nodes_list if n["type"] == "tank"]
        if not sources and nodes_list:
            sources = [nodes_list[0]["id"]]
    else:
        sources = pump_nodes
        
    # 인접 리스트 구축 (방향 없음)
    graph = {}
    for p in pipes_list:
        f, t = p["from"], p["to"]
        if f not in graph: graph[f] = []
        if t not in graph: graph[t] = []
        graph[f].append((t, p["id"]))
        graph[t].append((f, p["id"]))
        
    # BFS 탐색을 통해 소스(펌프)로부터의 노드 거리 및 방향성(DAG) 결정
    node_depth = {}
    queue = []
    for s in sources:
        node_depth[s] = 0
        queue.append(s)
        
    visited = set(sources)
    while queue:
        curr = queue.pop(0)
        curr_d = node_depth[curr]
        for neighbor, pipe_id in graph.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                node_depth[neighbor] = curr_d + 1
                queue.append(neighbor)
                
    # BFS 깊이에 따라 파이프의 방향을 from -> to 로 자동 Re-orient 정렬
    aligned_cnt = 0
    for p in pipes_list:
        f, t = p["from"], p["to"]
        
        # 🛡️ [물리학적 절대 법칙] 배관의 한쪽 끝이 TANK이고 다른 쪽이 PUMP인 흡입관(Suction Pipe)의 경우,
        # 유체는 무조건 TANK(공급 저장조) -> PUMP로 흘러들어가야 함!
        # BFS 위상 깊이 스케일과 상관없이, 강제로 from=TANK, to=PUMP로 위상 정렬 고정!
        f_type = node_map.get(f, {}).get("type", "")
        t_type = node_map.get(t, {}).get("type", "")
        
        if (f_type == "pump" and t_type == "tank"):
            p["from"] = t
            p["to"] = f
            aligned_cnt += 1
            continue
        elif (f_type == "tank" and t_type == "pump"):
            continue
            
        f_d = node_depth.get(f, 999)
        t_d = node_depth.get(t, 999)
        
        # 만약 to 노드가 from 노드보다 소스(펌프)에 더 가깝다면, 실제 유체 흐름은 t -> f 임.
        # 따라서 파이프의 방향성을 물리 흐름에 맞게 정렬 뒤집기!
        if t_d < f_d:
            p["from"] = t
            p["to"] = f
            aligned_cnt += 1
            
    if aligned_cnt > 0:
        print(f"위상 정렬 엔진: 총 {aligned_cnt}개의 배관 흐름 방향을 물리적 위상(펌프->아웃렛)에 맞게 정밀 동적 자동 보정 완료!")

def solve_pipe_network(pipes_list, nodes_list, rho, mu, epsilon, q_sys_lmin, material_name):
    # [A] 먼저 유동 위상 자동 정렬 엔진 가동하여 그리기 방향 무관 물리적 흐름 위상으로 재배치
    align_pipe_network_topology(pipes_list, nodes_list)
    
    node_map = {n["id"]: n for n in nodes_list}
    q_sys_m3s = float(q_sys_lmin) / 60000.0
    
    # [1] 마디(Junction/Node) 연속 방정식(질량 보존 법칙) 사전 적합성 진단
    in_flows = {}
    out_flows = {}
    for n_id in node_map:
        in_flows[n_id] = 0.0
        out_flows[n_id] = 0.0
        
    for p in pipes_list:
        f_n = p["from"]
        t_n = p["to"]
        q_val = float(p.get("Q", 0.0))
        out_flows[f_n] += q_val
        in_flows[t_n] += q_val
        
    continuity_warnings = []
    for n_id, node in node_map.items():
        if node["type"] in ["junction", "valve"]:
            diff = abs(in_flows[n_id] - out_flows[n_id])
            if diff > 1e-4:
                continuity_warnings.append(
                    f"마디 '{node['name']}' ({node['type'].upper()}): 유입량({in_flows[n_id]*60000:.1f} L/min)과 유출량({out_flows[n_id]*60000:.1f} L/min)의 불일치가 감지되었습니다. (질량 유동 오차: {diff*60000:.1f} L/min)"
                )
                
    st.session_state["continuity_warnings"] = continuity_warnings

    # [2] Spanning Tree(신장 트리) 기반 고도화된 독립 폐회로(Fundamental Loops) 자동 색출 알고리즘
    adj = {}
    for p in pipes_list:
        f, t = p["from"], p["to"]
        if f not in adj: adj[f] = []
        if t not in adj: adj[t] = []
        adj[f].append((t, p["id"]))
        adj[t].append((f, p["id"]))
    
    loops = []
    if pipes_list and nodes_list:
        visited = set()
        tree_edges = set()       # (u, v, pipe_id) 형태
        parent = {}              # node -> (parent_node, pipe_id)
        depth = {}               # node -> depth
        
        # 무방향 그래프 상의 모든 연결 성분(Connected Components)에 대해 Spanning Tree 구성
        for start_node in node_map:
            if start_node not in visited:
                queue = [start_node]
                visited.add(start_node)
                depth[start_node] = 0
                parent[start_node] = (None, None)
                
                while queue:
                    u = queue.pop(0)
                    for v, pipe_id in adj.get(u, []):
                        if v not in visited:
                            visited.add(v)
                            depth[v] = depth[u] + 1
                            parent[v] = (u, pipe_id)
                            # Tree edge 등록 (방향 무관하게 고유 키로)
                            n_min, n_max = min(u, v), max(u, v)
                            tree_edges.add((n_min, n_max, pipe_id))
                            queue.append(v)
        
        # 신장 트리에 포함되지 않은 나머지 파이프(Link / Co-tree edge)들을 순회하며 독립 루프 구성
        for p in pipes_list:
            u, v = p["from"], p["to"]
            n_min, n_max = min(u, v), max(u, v)
            
            # Tree Edge가 아닌 경우 정확히 하나의 독립 루프 형성
            if not any(e[2] == p["id"] for e in tree_edges):
                path_u = []
                path_v = []
                
                curr_u = u
                while curr_u is not None:
                    p_node, pipe_id = parent[curr_u]
                    if p_node is not None:
                        path_u.append((curr_u, p_node, pipe_id))
                    curr_u = p_node
                    
                curr_v = v
                while curr_v is not None:
                    p_node, pipe_id = parent[curr_v]
                    if p_node is not None:
                        path_v.append((curr_v, p_node, pipe_id))
                    curr_v = p_node
                
                # 공통 조상(LCA) 탐색
                path_u.reverse()
                path_v.reverse()
                
                i = 0
                min_len = min(len(path_u), len(path_v))
                while i < min_len and path_u[i][1] == path_v[i][1]:
                    i += 1
                
                # 🛡️ [컴포넌트 단절 세이프가드] 만약 루트 노드 자체가 다르면 분리된 그래프 성분이므로 루프 생성을 영구 회피!
                if i == 0 and (not path_u or not path_v or path_u[0][1] != path_v[0][1]):
                    continue
                
                u_branch = path_u[i:]
                v_branch = path_v[i:]
                u_branch.reverse() # u -> LCA 자식 방향으로
                
                loop = []
                
                # u에서 LCA 방향 경로 추가 (루프 내 시계방향을 가정)
                for child, par, pipe_id in u_branch:
                    loop.append((child, par, pipe_id, 1))
                
                # LCA에서 v 방향 경로 추가
                for child, par, pipe_id in v_branch:
                    loop.append((par, child, pipe_id, 1))
                    
                # Link 파이프 (v -> u)로 루프 완전 폐합
                loop.append((v, u, p["id"], 1))
                loops.append(loop)

    # -------------------------------------------------------------------------
    # 🌟 [자동 관경 추천 & 2단계 순환 하이브리드 수리해석 엔진 기동]
    # -------------------------------------------------------------------------
    original_Ds = {}
    for p in pipes_list:
        p_id = p["id"]
        original_Ds[p_id] = float(p.get("D", 0.08))
        if original_Ds[p_id] <= 0.005:
            p["D"] = 0.05
            
    # 단계 B. 1차 가지형 유량 순차적 배분(Branch Allocation) 및 초기 가정 유량 주입
    Q_1st = {}
    out_degree = {n_id: 0 for n_id in node_map}
    for p in pipes_list:
        out_degree[p["from"]] += 1
        
    pipe_minor_losses = {}
    for p in pipes_list:
        p_id = p["id"]
        k_val = 0.0
        for node_id in [p["from"], p["to"]]:
            if node_id in node_map:
                node_obj = node_map[node_id]
                if node_obj["type"] == "valve":
                    k_val += float(node_obj["val"])
        pipe_minor_losses[p_id] = k_val + 1.5

    visited_dist = set()
    def distribute_flow(node_id, current_flow):
        if node_id in visited_dist:
            return
        visited_dist.add(node_id)
        
        pipes_from = [p for p in pipes_list if p["from"] == node_id]
        if not pipes_from:
            return
        
        # 🛡️ [수력 저항 비례 유량 분배 모델] 균등 배분 대신 실제 배관 마찰 저항 K에 따라 유량 비례 배분
        K_ests = []
        for p in pipes_from:
            d_m = float(p.get("D", 0.08))
            l_m = float(p.get("L", 10.0))
            f_est = 0.02 # 임의 가상의 난류 마찰계수
            k_minor = pipe_minor_losses.get(p["id"], 1.5)
            # 수력 저항 계수 산출 (Darcy-Weisbach K)
            K_est = (f_est * (l_m / d_m) + k_minor) / (2.0 * 9.81 * (np.pi/4.0 * d_m**2)**2)
            K_ests.append(max(K_est, 1e-3))
            
        inv_sqrt_sum = sum(1.0 / np.sqrt(k) for k in K_ests)
        
        for idx, p in enumerate(pipes_from):
            p_id = p["id"]
            weight = (1.0 / np.sqrt(K_ests[idx])) / inv_sqrt_sum
            flow_share = current_flow * weight
            Q_1st[p_id] = flow_share
            distribute_flow(p["to"], flow_share)
            
    pump_nodes = [n["id"] for n in nodes_list if n["type"] == "pump"]
    main_source = pump_nodes[0] if pump_nodes else (nodes_list[0]["id"] if nodes_list else None)
    
    if main_source:
        distribute_flow(main_source, q_sys_m3s)
        
    for p in pipes_list:
        p_id = p["id"]
        if p_id not in Q_1st or Q_1st[p_id] <= 1e-9:
            user_q = float(p.get("Q", 0.0))
            Q_1st[p_id] = user_q if user_q > 0 else q_sys_m3s / max(len(pipes_list), 1)

    max_iter = 150
    tol = 1e-6
    
    pump_shutoffs = {}
    for n in nodes_list:
        if n["type"] == "pump":
            pump_shutoffs[n["id"]] = float(n["val"]) if float(n["val"]) > 0 else 50.0

    # 1차 루프 연산 (Signed Flow 하디크로스)
    if loops:
        for iteration in range(max_iter):
            max_delta = 0.0
            for loop in loops:
                sum_h = 0.0
                sum_dq = 0.0
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    
                    # 루프 순회 방향과 파이프 고유 방향(from->to) 일치 여부 판정
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    d_m = float(p_obj["D"])
                    l_m = float(p_obj["L"])
                    q_val = Q_1st[pipe_id]
                    
                    # 루프 순회 기준 부호 있는 유량
                    q_loop = q_val * sgn_loop
                    abs_q = abs(q_loop)
                    
                    v_flow = calc_velocity(abs_q, d_m)
                    re = calc_reynolds(rho, v_flow, d_m, mu)
                    f, _ = calc_friction_factor(re, d_m, epsilon)
                    
                    g = 9.81
                    # K 저항 계수 계산 (Darcy-Weisbach)
                    K_dw = (f * (l_m / d_m) + pipe_minor_losses[pipe_id]) / (2.0 * g * (np.pi/4.0 * d_m**2)**2)
                    
                    # 수두 손실 = K * Q * |Q|
                    h_loss = K_dw * q_loop * abs_q
                    
                    # 펌프 모델링 (노드 기준의 중복 가속을 해결하기 위해 해당 펌프 유로관 단위로 가압 보정)
                    h_pump = 0.0
                    A_coeff = 50000.0
                    is_pump_pipe = (p_obj["from"] in pump_shutoffs)
                    
                    if is_pump_pipe:
                        # 펌프는 순방향 유동(q_val > 0)일 때만 유효 가압 양정을 작용하며 역방향 시 가압 없음
                        if q_val > 0.0:
                            h_pump = max(pump_shutoffs[p_obj["from"]] - A_coeff * (q_val ** 2), 0.0)
                        else:
                            h_pump = 0.0
                        
                    sum_h += (h_loss - h_pump * sgn_loop)
                    
                    # 수치 안정성 가드 상수가 가미된 미분항 (n * K * |Q|)
                    sum_dq += (2.0 * K_dw * abs_q + (2.0 * A_coeff * abs_q if h_pump > 0.0 else 0.0) + 1e-5)
                        
                sum_dq = max(sum_dq, 1e-4)
                delta_q = - sum_h / sum_dq
                
                # 🛡️ [수치 해석 이완 완충 계수 도입] 인접 루프 간 유동 진동 방지를 위해 0.65 이완율 도입
                delta_q = delta_q * 0.65
                
                max_delta = max(max_delta, abs(delta_q))
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    # 각 파이프 유량 보정 (부호 있는 방향성 반영)
                    Q_1st[pipe_id] += delta_q * sgn_loop
            if max_delta < tol:
                break

    # 단계 C. 1차 수렴 유량 결과를 바탕으로, 각 파이프의 최적 추천 직경 산정 및 자동 굵기 대입
    optimal_specs = {}
    for p in pipes_list:
        p_id = p["id"]
        q_calc = abs(Q_1st[p_id])
        l_m = float(p["L"])
        
        # 경제적 추천 내경 및 KS 규격명 획득
        rec_d, rec_spec = recommend_pipe_spec(q_calc, material_name)
        
        # 🛡️ [물리학적 절대 안전] 펌프 흡입 배관인 경우 NPSH 공동현상이 절대 나지 않도록 구경 강제 역산 보정
        is_suction_pipe = False
        to_node_id = p["to"]
        if to_node_id in node_map and node_map[to_node_id]["type"] == "pump":
            is_suction_pipe = True
            
        if is_suction_pipe:
            try:
                # 세션 스토리지에서 현재 온도와 유체 증기압 조회
                temp_val = st.session_state.get("shared_temp", 20.0)
                # 세션에 shared_fluid가 한글명일 경우 키를 직접 역산
                fluid_display = st.session_state.get("shared_fluid", "물 (Water)")
                fluid_k = "Water"
                for k, v in FLUID_OPTIONS.items():
                    if v == fluid_display:
                        fluid_k = k
                        break
                _, _, p_vap_calc = get_fluid_properties(fluid_k, temp_val)
            except Exception:
                p_vap_calc = 2300.0  # 물의 상온 기준 포화증기압 폴백
                
            g_c = 9.81
            h_atm = 101325.0 / (rho * g_c)
            h_vap = p_vap_calc / (rho * g_c)
            
            # 유량에 따른 NPSHr 추정
            q_m3h = (q_calc * 3600.0)
            npshr_est = 2.0
            if q_m3h > 4.0: npshr_est = 2.5
            if q_m3h > 8.0: npshr_est = 3.0
            if q_m3h > 16.0: npshr_est = 3.5
            
            target_npsha = npshr_est + 0.5  # 최소 안전 마진 0.5m 확보
            
            # 수치 해석법으로 NPSHa >= target_npsha를 만족하는 안전 최소 내경 D_npsh 산출
            d_npsh = 0.015
            for test_d_mm in range(15, 200, 5):
                test_d = test_d_mm / 1000.0
                test_v = calc_velocity(q_calc, test_d)
                test_re = calc_reynolds(rho, test_v, test_d, mu)
                test_f, _ = calc_friction_factor(test_re, test_d, epsilon)
                test_h_fs = (test_f * (l_m / test_d) + 1.5) * (test_v**2) / (2.0 * g_c)
                test_npsha = h_atm - h_vap - test_h_fs
                if test_npsha >= target_npsha:
                    d_npsh = test_d
                    break
            
            # 만약 경제 관경(rec_d)이 NPSH 안전 최소 관경(d_npsh)에 못 미치면, 안전을 위해 강제 격상!
            if rec_d < d_npsh:
                rec_d = d_npsh
                # recommend_pipe_spec에 안전 최소경 min_d_m 가이드라인 제공하여 규격명과 내경 수치 물리적 일치 확보!
                rec_d, rec_spec = recommend_pipe_spec(q_calc, material_name, min_d_m=d_npsh)
                
        optimal_specs[p_id] = {"D": rec_d, "spec": rec_spec}
        p["t_rec"] = rec_spec
        
        if original_Ds[p_id] <= 0.005 or abs(original_Ds[p_id] - 0.08) < 1e-4 or abs(original_Ds[p_id] - 0.1) < 1e-4:
            p["D"] = rec_d

    # 단계 D. 2차 최종 하디크로스 수리해석 기동 (업데이트된 최적 직경 세트 기준)
    Q_2nd = {}
    for p in pipes_list:
        p_id = p["id"]
        Q_2nd[p_id] = Q_1st[p_id]

    # 단계 E. 펌프 소요 양정(H) 자동 역산 (비재귀 BFS 기반 고도화된 Critical Path 손실 수두 계산)
    path_head_losses = {n["id"]: 0.0 for n in nodes_list}
    if main_source:
        queue = [(main_source, 0.0)]
        visited_trace = set()
        
        while queue:
            curr_node, acc_loss = queue.pop(0)
            if acc_loss > path_head_losses.get(curr_node, 0.0):
                path_head_losses[curr_node] = acc_loss
                
            if curr_node in visited_trace:
                continue
            visited_trace.add(curr_node)
            
            pipes_from = [p for p in pipes_list if p["from"] == curr_node]
            for p in pipes_from:
                p_id = p["id"]
                d_m = float(p["D"])
                l_m = float(p["L"])
                q_val = abs(Q_2nd.get(p_id, 1e-4))
                
                v_flow = calc_velocity(q_val, d_m)
                re = calc_reynolds(rho, v_flow, d_m, mu)
                f, _ = calc_friction_factor(re, d_m, epsilon)
                
                g = 9.81
                h_fric = f * (l_m / d_m) * (v_flow**2) / (2 * g)
                h_minor = pipe_minor_losses.get(p_id, 1.5) * (v_flow**2) / (2 * g)
                p_loss_head = h_fric + h_minor
                
                next_node = p["to"]
                queue.append((next_node, acc_loss + p_loss_head))
        
    total_loss_head = max(path_head_losses.values()) if path_head_losses else 5.0
    calculated_pump_head = max(total_loss_head + 10.0, 15.0)
    
    pump_shutoffs_final = {}
    for n in nodes_list:
        if n["type"] == "pump":
            user_h = float(n["val"])
            if user_h <= 0.5:
                n["val"] = round(calculated_pump_head, 1)
                pump_shutoffs_final[n["id"]] = calculated_pump_head
            else:
                pump_shutoffs_final[n["id"]] = user_h

    # 최종 하디크로스 수렴 가동
    if loops:
        for iteration in range(max_iter):
            max_delta = 0.0
            for loop in loops:
                sum_h = 0.0
                sum_dq = 0.0
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    d_m = float(p_obj["D"])
                    l_m = float(p_obj["L"])
                    q_val = Q_2nd[pipe_id]
                    
                    q_loop = q_val * sgn_loop
                    abs_q = abs(q_loop)
                    
                    v_flow = calc_velocity(abs_q, d_m)
                    re = calc_reynolds(rho, v_flow, d_m, mu)
                    f, _ = calc_friction_factor(re, d_m, epsilon)
                    
                    g = 9.81
                    K_dw = (f * (l_m / d_m) + pipe_minor_losses[pipe_id]) / (2.0 * g * (np.pi/4.0 * d_m**2)**2)
                    
                    h_loss = K_dw * q_loop * abs_q
                    
                    h_pump = 0.0
                    A_coeff = 50000.0
                    is_pump_pipe = (p_obj["from"] in pump_shutoffs_final)
                    
                    if is_pump_pipe:
                        if q_val > 0.0:
                            h_pump = max(pump_shutoffs_final[p_obj["from"]] - A_coeff * (q_val ** 2), 0.0)
                        else:
                            h_pump = 0.0
                        
                    sum_h += (h_loss - h_pump * sgn_loop)
                    sum_dq += (2.0 * K_dw * abs_q + (2.0 * A_coeff * abs_q if h_pump > 0.0 else 0.0) + 1e-5)
                        
                sum_dq = max(sum_dq, 1e-4)
                delta_q = - sum_h / sum_dq
                
                # 수치 해석 완충 이완계수 0.65 곱셈 가중
                delta_q = delta_q * 0.65
                
                max_delta = max(max_delta, abs(delta_q))
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    Q_2nd[pipe_id] += delta_q * sgn_loop
            if max_delta < tol:
                break
                
    for p in pipes_list:
        p_id = p["id"]
        q_final = Q_2nd.get(p_id, 0.0)
        if np.isnan(q_final) or np.isinf(q_final):
            q_final = 0.0
        d_m = float(p.get("D", 0.08))
        if np.isnan(d_m) or np.isinf(d_m) or d_m <= 0.0:
            d_m = 0.08
        v_flow = calc_velocity(abs(q_final), d_m)
        if np.isnan(v_flow) or np.isinf(v_flow):
            v_flow = 0.0
        p["Q"] = float(q_final)
        p["v_flow"] = float(v_flow)

    # 🛡️ 수격, 처짐 및 서포터 연산 중 혹은 하디크로스 도중 발생할 수 있는 모든 NaN/Infinity 최종 세척 소독!
    clean_numerical_results(pipes_list, nodes_list)

    return Q_2nd

def clean_numerical_results(pipes_list, nodes_list):
    for p in pipes_list:
        for k, v in list(p.items()):
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    p[k] = 0.0
            elif isinstance(v, str):
                if v.lower() in ["nan", "inf", "infinity", "-inf", "-infinity"]:
                    p[k] = ""
                    
    for n in nodes_list:
        for k, v in list(n.items()):
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    n[k] = 0.0
            elif isinstance(v, str):
                if v.lower() in ["nan", "inf", "infinity", "-inf", "-infinity"]:
                    n[k] = ""


ROUGHNESS = {
    "Smooth Pipe (초매끈한 관, ε=0)": 0.0,
    "PVC (일반 플라스틱 관)": 1.5e-6,
    "Commercial Steel (상업용 강관)": 4.6e-5,
    "Galvanized Steel (아연도금 강관)": 1.5e-4,
    "Cast Iron (주철관)": 2.6e-4,
    "Concrete (콘크리트관)": 1.5e-3,
    "Drawn Tubing (인발 튜브)": 1.5e-6,
    "Stainless Steel (스테인리스 강관)": 1.5e-5,
}

MECHANICAL_PROPS = {
    "Smooth Pipe (초매끈한 관, ε=0)": {"E": 3e9, "alpha": 5e-5, "Sy": 45e6},
    "PVC (일반 플라스틱 관)": {"E": 3e9, "alpha": 5e-5, "Sy": 45e6},
    "Commercial Steel (상업용 강관)": {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6},
    "Galvanized Steel (아연도금 강관)": {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6},
    "Cast Iron (주철관)": {"E": 100e9, "alpha": 1.04e-5, "Sy": 200e6},
    "Concrete (콘크리트관)": {"E": 25e9, "alpha": 1.0e-5, "Sy": 5e6}, 
    "Drawn Tubing (인발 튜브)": {"E": 100e9, "alpha": 1.5e-5, "Sy": 150e6},
    "Stainless Steel (스테인리스 강관)": {"E": 193e9, "alpha": 1.6e-5, "Sy": 205e6},
}

FITTING_LOSSES = {
    "90도 엘보우 (Standard)": 0.75,
    "45도 엘보우 (Standard)": 0.40,
    "티 (Straight run)": 0.60,
    "티 (Branch flow)": 1.80,
}
VALVE_LOSSES = {
    "게이트 밸브 (Fully open)": 0.15,
    "글로브 밸브 (Fully open)": 10.0,
    "스윙 체크 밸브": 2.0,
}

PIPE_STANDARDS = {
    "KS D 3576 (스테인리스 강관)": {
        "schedules": ["SCH 5S", "SCH 10S", "SCH 20S", "SCH 40", "SCH 80"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.8, "SCH 80": 3.7},
            "20A (3/4B)":   {"OD": 27.2, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.9, "SCH 80": 3.9},
            "25A (1B)":     {"OD": 34.0, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.4, "SCH 80": 4.5},
            "32A (1 1/4B)": {"OD": 42.7, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.6, "SCH 80": 4.9},
            "40A (1 1/2B)": {"OD": 48.6, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.7, "SCH 80": 5.1},
            "50A (2B)":     {"OD": 60.5, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.5, "SCH 40": 3.9, "SCH 80": 5.5},
            "65A (2 1/2B)": {"OD": 76.3, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 3.5, "SCH 40": 5.2, "SCH 80": 7.0},
            "80A (3B)":     {"OD": 89.1, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.5, "SCH 80": 7.6},
            "90A (3 1/2B)": {"OD": 101.6,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.7, "SCH 80": 8.1},
            "100A (4B)":    {"OD": 114.3,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 6.0, "SCH 80": 8.6},
            "125A (5B)":    {"OD": 139.8,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 6.6, "SCH 80": 9.5},
            "150A (6B)":    {"OD": 165.2,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 7.1, "SCH 80": 11.0},
            "200A (8B)":    {"OD": 216.3,"SCH 5S": 2.8,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 8.2, "SCH 80": 12.7},
            "250A (10B)":   {"OD": 267.4,"SCH 5S": 3.4,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 9.3, "SCH 80": 15.1},
            "300A (12B)":   {"OD": 318.5,"SCH 40": 4.0,  "SCH 10S": 4.5, "SCH 20S": 6.5, "SCH 40": 10.3,"SCH 80": 17.4},
        }
    },
    "KS D 3507 (일반 배관용 탄소강관)": {
        "schedules": ["일반배관 (SPP)"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "일반배관 (SPP)": 2.8},
            "20A (3/4B)":   {"OD": 27.2, "일반배관 (SPP)": 2.8},
            "25A (1B)":     {"OD": 34.0, "일반배관 (SPP)": 3.2},
            "32A (1 1/4B)": {"OD": 42.7, "일반배관 (SPP)": 3.5},
            "40A (1 1/2B)": {"OD": 48.6, "일반배관 (SPP)": 3.5},
            "50A (2B)":     {"OD": 60.5, "일반배관 (SPP)": 3.8},
            "65A (2 1/2B)": {"OD": 76.3, "일반배관 (SPP)": 4.2},
            "80A (3B)":     {"OD": 89.1, "일반배관 (SPP)": 4.2},
            "100A (4B)":    {"OD": 114.3,"일반배관 (SPP)": 4.5},
            "125A (5B)":    {"OD": 139.8,"일반배관 (SPP)": 4.5},
            "150A (6B)":    {"OD": 165.2,"일반배관 (SPP)": 5.0},
            "200A (8B)":    {"OD": 216.3,"일반배관 (SPP)": 5.8},
            "250A (10B)":   {"OD": 267.4,"일반배관 (SPP)": 6.6},
            "300A (12B)":   {"OD": 318.5,"일반배관 (SPP)": 6.9},
        }
    },
    "KS D 3562 (압력 배관용 탄소강관)": {
        "schedules": ["SCH 40", "SCH 80", "SCH 160"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "SCH 40": 2.8, "SCH 80": 3.7, "SCH 160": 4.7},
            "20A (3/4B)":   {"OD": 27.2, "SCH 40": 2.9, "SCH 80": 3.9, "SCH 160": 5.5},
            "25A (1B)":     {"OD": 34.0, "SCH 40": 3.4, "SCH 80": 4.5, "SCH 160": 6.4},
            "32A (1 1/4B)": {"OD": 42.7, "SCH 40": 3.6, "SCH 80": 4.9, "SCH 160": 6.4},
            "40A (1 1/2B)": {"OD": 48.6, "SCH 40": 3.7, "SCH 80": 5.1, "SCH 160": 7.1},
            "50A (2B)":     {"OD": 60.5, "SCH 40": 3.9, "SCH 80": 5.5, "SCH 160": 8.7},
            "65A (2 1/2B)": {"OD": 76.3, "SCH 40": 5.2, "SCH 80": 7.0, "SCH 160": 9.5},
            "80A (3B)":     {"OD": 89.1, "SCH 40": 5.5, "SCH 80": 7.6, "SCH 160": 11.1},
            "100A (4B)":    {"OD": 114.3,"SCH 40": 6.0, "SCH 80": 8.6, "SCH 160": 13.5},
            "125A (5B)":    {"OD": 139.8,"SCH 40": 6.6, "SCH 80": 9.5, "SCH 160": 15.9},
            "150A (6B)":    {"OD": 165.2,"SCH 40": 7.1, "SCH 80": 11.0,"SCH 160": 18.2},
            "200A (8B)":    {"OD": 216.3,"SCH 40": 8.2, "SCH 80": 12.7,"SCH 160": 23.0},
            "250A (10B)":   {"OD": 267.4,"SCH 40": 9.3, "SCH 80": 15.1,"SCH 160": 28.6},
            "300A (12B)":   {"OD": 318.5,"SCH 40": 10.3,"SCH 80": 17.4,"SCH 160": 33.3},
        }
    }
}

from functools import lru_cache

@lru_cache(maxsize=256)
def get_fluid_properties(fluid: str, temp_c: float) -> tuple:
    T_K = temp_c + 273.15
    P = 101325.0
    p_vapor = 0.0
    
    if fluid.startswith("INCOMP::"):
        rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
        mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
        p_vapor = 2000.0
    else:
        try:
            rho = CP.PropsSI("D", "T", T_K, "Q", 0, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "Q", 0, fluid)
            p_vapor = CP.PropsSI("P", "T", T_K, "Q", 0, fluid)
        except ValueError:
            try:
                # 🛡️ [열역학적 절대 안정가드] 끓는점 초과 실패 시 기화로 밀도가 급락하지 않도록 
                # 강제 상온(20도) 액체 상으로 물성을 조회하여 레이놀즈수 대격변을 차단
                T_safe_K = min(T_K, 20.0 + 273.15)
                rho = CP.PropsSI("D", "T", T_safe_K, "Q", 0, fluid)
                mu  = CP.PropsSI("V", "T", T_safe_K, "Q", 0, fluid)
                p_vapor = CP.PropsSI("P", "T", T_safe_K, "Q", 0, fluid)
            except Exception:
                rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
                mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
                p_vapor = P
                if rho < 500.0:  # 극도로 밀도가 떨어지는 기체상의 경우 20도 표준액 풀백
                    rho = 998.2
                    mu = 0.001002
                    p_vapor = 2300.0
            
    return rho, mu, p_vapor

def calc_velocity(Q_m3s: float, D: float) -> float:
    if np.isnan(Q_m3s) or np.isinf(Q_m3s): Q_m3s = 0.0
    if np.isnan(D) or np.isinf(D) or D <= 0: D = 0.08
    D = max(D, 1e-9)
    A = np.pi / 4.0 * D**2
    return Q_m3s / A

def calc_reynolds(rho: float, v: float, D: float, mu: float) -> float:
    if np.isnan(v) or np.isinf(v): v = 0.0
    if np.isnan(rho) or np.isinf(rho) or rho <= 0: rho = 998.2
    if np.isnan(D) or np.isinf(D) or D <= 0: D = 0.08
    if np.isnan(mu) or np.isinf(mu) or mu <= 0: return float('inf')
    return (rho * v * D) / mu

def calc_friction_factor(Re: float, D: float, epsilon: float) -> tuple:
    if np.isnan(Re) or np.isinf(Re) or Re < 1e-6:
        return 0.0, "정지 (No Flow)"
    if np.isnan(D) or np.isinf(D) or D <= 0: D = 0.08
    if np.isnan(epsilon) or np.isinf(epsilon) or epsilon < 0: epsilon = 4.6e-5
    
    if Re < 2300:
        f = 64.0 / Re
        regime = "층류 (Laminar)"
    elif Re < 4000:
        # 천이 영역 (Transitional Zone): 층류와 난류 마찰계수를 smoothstep 보간하여 불연속성 극복
        D = max(D, 1e-9)
        f_lam = 64.0 / 2300.0
        rel_rough = epsilon / D
        denom_4000 = np.log10(max(rel_rough / 3.7 + 5.74 / (4000.0**0.9), 1e-15))
        f_turb = 0.25 / denom_4000**2
        
        # 보간 가중치 t 및 smoothstep 매핑
        t = (Re - 2300.0) / (4000.0 - 2300.0)
        h = t * t * (3.0 - 2.0 * t)
        f = f_lam + h * (f_turb - f_lam)
        regime = "전이구간 (Transitional)"
    else:
        D = max(D, 1e-9)
        relative_roughness = epsilon / D
        denom = np.log10(max(relative_roughness / 3.7 + 5.74 / (Re**0.9), 1e-15))
        f = 0.25 / denom**2
        regime = "난류 (Turbulent)"
        
    if np.isnan(f) or np.isinf(f):
        f = 0.02
        
    return f, regime

def calc_pressure_dp(f: float, L: float, D: float, rho: float, v: float, sum_K_fit: float, sum_K_valve: float) -> tuple:
    D = max(D, 1e-9)
    dynamic_pressure = rho * v**2 / 2.0
    dp_fric = f * (L / D) * dynamic_pressure
    dp_fit = sum_K_fit * dynamic_pressure
    dp_valve = sum_K_valve * dynamic_pressure
    dp_total = dp_fric + dp_fit + dp_valve
    return dp_fric, dp_fit, dp_valve, dp_total

def calc_pump_power(dp_pa: float, Q_m3s: float, eff: float) -> float:
    if eff <= 0: return 0.0
    power_watts = (dp_pa * Q_m3s) / (eff / 100.0)
    return power_watts / 1000.0

def get_standard_motor(kw_req: float) -> tuple:
    std_sizes = [0.4, 0.75, 1.5, 2.2, 3.7, 5.5, 7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0, 55.0, 75.0, 90.0, 110.0, 132.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0]
    design_kw = kw_req * 1.15
    for size in std_sizes:
        if size >= design_kw:
            return size, f"효성 프리미엄 고효율 전동기(IE3) / {size}kW 급"
    return design_kw, f"초대형 맞춤 제작 전동기 / {design_kw:.1f}kW 급"

def get_recommended_thickness_ks(outer_d_mm: float, req_t_mm: float, material_type: str) -> tuple:
    std_key = "KS D 3576 (스테인리스 강관)" if "Stainless" in material_type else "KS D 3562 (압력 배관용 탄소강관)"
    std_data = PIPE_STANDARDS[std_key]
    
    closest_nps = None
    min_diff = float('inf')
    
    for nps, info in std_data["data"].items():
        diff = abs(info["OD"] - outer_d_mm)
        if diff < min_diff:
            min_diff = diff
            closest_nps = nps
            
    if closest_nps is None:
        return "N/A", req_t_mm
        
    nps_info = std_data["data"][closest_nps]
    schedules = [sch for sch in std_data["schedules"] if sch in nps_info]
    
    recommended_sch = "N/A"
    recommended_t = 0.0
    
    for sch in schedules:
        t_sch = nps_info[sch]
        if t_sch >= req_t_mm:
            recommended_sch = sch
            recommended_t = t_sch
            break
            
    if recommended_sch == "N/A" and len(schedules) > 0:
        recommended_sch = schedules[-1] + " (두께 부족, 외경 상향 권장)"
        recommended_t = nps_info[schedules[-1]]
        
    return f"{closest_nps} - {recommended_sch}", recommended_t

