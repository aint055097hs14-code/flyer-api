import base64, json, random, re, os, mimetypes
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

# 註冊 .glb / .gltf 的正確 MIME type
# (Railway 的 Linux 環境預設 mimetypes 資料庫不認得這兩種)
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

app = FastAPI(title="售屋傳單解析 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 立面零件靜態檔案路由
# /parts/part_00.glb ~ /parts/part_15.glb
if os.path.isdir("parts"):
    app.mount("/parts", StaticFiles(directory="parts"), name="parts")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# ── 模型外觀描述庫 ────────────────────────────────────────────
PART_DESC = {
    0:  {"label":"磚紅鐵窗老公寓",      "feel":"老舊居住痕跡濃厚，時代感的住宅立面"},
    1:  {"label":"深磚商業招牌牆",      "feel":"街道商業氣息強烈，混合居住與商業功能"},
    2:  {"label":"水泥素牆小鐵窗",      "feel":"極度老化低調壓抑，接近廢棄邊緣的存在感"},
    3:  {"label":"暗磚老化鐵捲門",      "feel":"嚴重風化，透露出被遺忘與待修繕的滄桑"},
    4:  {"label":"白格窗磁磚雨遮",      "feel":"老式住宅的日常生活感，帶有植物與生活痕跡"},
    5:  {"label":"白窗磚牆商住複合",    "feel":"商業與住宅混用，街道生活機能感強"},
    6:  {"label":"深灰素面倉庫牆",      "feel":"工業冷調，極簡封閉，拒絕外部視線"},
    7:  {"label":"磚底騎樓柱廊",        "feel":"台灣街屋底層原型，承載公共通道與私人邊界"},
    8:  {"label":"白牆弧拱日治遺風",    "feel":"日治時代建築語彙，帶有殖民地現代性的歷史感"},
    9:  {"label":"淡藍波浪板違建",      "feel":"非正式加建，臨時性與生存策略的空間表達"},
    10: {"label":"白灰格窗素面住宅",    "feel":"無裝飾的日常住宅，低調融入城市背景"},
    11: {"label":"藍鐵皮頂加結構",      "feel":"頂層非法加建，城市高密度生存的即興建築"},
    12: {"label":"綠鐵皮陽台頂層",      "feel":"簡易加建的生活空間，自力更生的居住策略"},
    13: {"label":"紅磚弧拱巴洛克屋頂",  "feel":"南歐或日治巴洛克裝飾語彙，強調歷史與身份地位"},
    14: {"label":"粉磚透天住宅一樓",    "feel":"典型台灣透天厝，家族居住記憶與土地所有感"},
    15: {"label":"板岩藍門騎樓底層",    "feel":"混搭材質的商業底層，呈現台北街道的拼貼美學"},
}
PART_ROLE = {
    "ground_floor": "一樓（街道介面）",
    "middle":       "中間層（主要居住面）",
    "ring":         "環狀層（立面節奏）",
    "highfloor":    "高樓層（天際線輪廓）",
    "rooftop":      "屋頂（頂部收尾）",
}

# ── 機器視野四維評分 ──────────────────────────────────────────
# BAI Boundary Anxiety Index      邊界焦慮指數   → Eisenman 解構／切割
# END Entropy Node Density        熵節點密度     → Eisenman 錯位／重疊
# SSI Structural Self-evidence    結構自明性指數 → Eisenman 摺疊／造型語法
# AEI Axial Escape Index          軸向逸脫指數   → Eisenman 旋轉／系統偏移
PART_MACHINE = {
    0:  {"bai":7, "end":5, "ssi":5, "aei":3},
    1:  {"bai":9, "end":9, "ssi":3, "aei":9},
    2:  {"bai":2, "end":2, "ssi":6, "aei":1},
    3:  {"bai":3, "end":4, "ssi":4, "aei":3},
    4:  {"bai":7, "end":7, "ssi":5, "aei":4},
    5:  {"bai":5, "end":3, "ssi":7, "aei":2},
    6:  {"bai":6, "end":6, "ssi":5, "aei":4},
    7:  {"bai":8, "end":4, "ssi":9, "aei":2},
    8:  {"bai":8, "end":2, "ssi":9, "aei":2},
    9:  {"bai":4, "end":6, "ssi":1, "aei":8},
    10: {"bai":5, "end":3, "ssi":6, "aei":2},
    11: {"bai":7, "end":9, "ssi":1, "aei":9},
    12: {"bai":6, "end":7, "ssi":2, "aei":7},
    13: {"bai":9, "end":3, "ssi":9, "aei":3},
    14: {"bai":6, "end":5, "ssi":7, "aei":4},
    15: {"bai":7, "end":6, "ssi":7, "aei":5},
}

def calc_machine_scores(facade: dict, params: dict) -> dict:
    h  = params.get("height", 4)
    rf = params.get("rings_frequency_h", 4)
    uf = params.get("upper_floors", 0)
    counts = {
        "ground_floor": 1,
        "middle":       h,
        "ring":         max(1, round(h / rf)),
        "highfloor":    uf,
        "rooftop":      1,
    }
    totals = {"bai":0,"end":0,"ssi":0,"aei":0}
    breakdown = {}
    for part, count in counts.items():
        if count == 0: continue
        t_num = facade.get(part, {}).get("true", 0)
        f_num = facade.get(part, {}).get("false", 0)
        t_score = PART_MACHINE.get(t_num, {"bai":5,"end":5,"ssi":5,"aei":5})
        f_score = PART_MACHINE.get(f_num, {"bai":5,"end":5,"ssi":5,"aei":5})
        for k in ["bai","end","ssi","aei"]:
            totals[k] += round((t_score[k]+f_score[k])/2 * count)
        breakdown[part] = {"count":count,"unit_t":t_num,"unit_f":f_num}
    bai_penalty   = max(0, totals["bai"] - 80)
    ssi_effective = max(0, totals["ssi"] - bai_penalty)
    overload_risk = totals["end"] + totals["aei"]
    preserved     = ssi_effective >= 70
    return {
        "bai": totals["bai"],
        "end": totals["end"],
        "ssi": totals["ssi"],
        "aei": totals["aei"],
        "ssi_effective": ssi_effective,
        "overload_risk": overload_risk,
        "preserved":     preserved,
        "breakdown":     breakdown,
        "eisenman": {
            "decomposition":   round(totals["bai"]/10, 1),
            "superimposition": max(1, round(totals["end"]/10*4)),
            "self_evidence":   round(ssi_effective/10, 1),
            "axial_deviation": round(totals["aei"]/10, 1),
        }
    }

def generate_interpretation(record, params, facade, meta):
    lines = []
    tier_map = {"low":"低正向（廣告語言模糊、承諾有限）",
                "mid":"中正向（一般市場描述）",
                "high":"高正向（強力承諾居住品質與價值）"}
    tier_label = tier_map.get(meta.get("sentiment_tier",""), "")
    cluster = meta.get("primary_cluster","")
    conf    = meta.get("cluster_confidence", 0)
    prop_type = record.get("property_type") or "物件"
    price     = record.get("price_wan") or "？"
    area      = record.get("area_ping") or "？"
    loc       = record.get("location_desc") or record.get("address") or "台北"
    reason    = meta.get("cluster_reason","")
    lines.append(
        f"這棟建築由一張{prop_type}傳單生成，售價 {price} 萬，坪數 {area} 坪，位於{loc}。\n"
        f"傳單廣告情感屬於{tier_label}，語意導向判定為「{cluster}」（信心 {conf}%）。"
        + (f"「{reason}」" if reason else "")
    )
    part_lines = []
    for part_key, part_label in PART_ROLE.items():
        if part_key not in facade: continue
        t_num = facade[part_key].get("true")
        f_num = facade[part_key].get("false")
        t = PART_DESC.get(t_num, {"label":f"#{t_num}","feel":""})
        f = PART_DESC.get(f_num, {"label":f"#{f_num}","feel":""})
        if t_num == f_num:
            part_lines.append(f"・{part_label}：【{t['label']}】— {t['feel']}")
        else:
            part_lines.append(
                f"・{part_label}：【{t['label']}】與【{f['label']}】之間的交替，"
                f"呈現{t['feel']}，對照{f['feel']}")
    lines.append("建築各部位的外觀邏輯：\n" + "\n".join(part_lines))
    h = params.get("height", 4)
    w = params.get("width", 2)
    hd = "低矮" if h<=4 else ("中等" if h<=6 else "高聳")
    wd = "窄小" if w<=2 else ("中型" if w<=4 else "寬闊")
    lines.append(
        f"整體而言，這是一棟{hd}的{wd}建築，以台北真實街道的立面碎片拼湊而成。"
        "它既不是廣告傳單所承諾的理想居所，也不是城市規劃圖上的標準單元——"
        "而是將房產買賣的語言轉譯成一種關於台北城市紋理的建築想像。"
    )
    return "\n\n".join(lines)


# ── 立面模型池 ────────────────────────────────────────────────
SENTIMENT_POOLS = {
    "low":  {"middle":[1,2,3,12,14], "ring":[1,2,3,12,14], "highfloor":[1,2,3,12,14]},
    "mid":  {"middle":[0,10,12],     "ring":[0,1,10,12,14],"highfloor":[0,10,12]},
    "high": {"middle":[0,8,9],       "ring":[0,8,9,10],    "highfloor":[0,8,13]},
}
SENTIMENT_DEFAULT = {
    "middle":    {"true":0, "false":[0,12]},
    "ring":      {"true":1, "false":[0,2,12]},
    "highfloor": {"true":0, "false":[0,12]},
}
CLUSTER_POOLS = {
    "交通導向": {"ground_true":[7,13,15],  "ground_false":[0,5,6,7,13,15], "roof_true":[12,14],     "roof_false":[12,14]},
    "機能導向": {"ground_true":[9,11],     "ground_false":[0,5,6,7,9,11],  "roof_true":[12,14],     "roof_false":[12,14]},
    "投資導向": {"ground_true":[13],       "ground_false":[0,13],           "roof_true":[10],        "roof_false":[10]},
    "景觀導向": {"ground_true":[13,9],     "ground_false":[0,9],            "roof_true":[10,13,14],  "roof_false":[10,13,14]},
}
CLUSTER_DEFAULT = {"ground_true":[7],"ground_false":[0],"roof_true":[14],"roof_false":[14]}

def pick(rng, pool):
    if isinstance(pool, list): return rng.choice(pool) if pool else 0
    return pool

def resolve_tier(s):
    if s is None: return None
    return "low" if s<=0.5 else ("mid" if s<=0.8 else "high")

def build_facade(sentiment_score, primary_cluster, seed: int):
    rng = random.Random(seed)   # 固定 seed → 固定結果
    tier = resolve_tier(sentiment_score)
    result = {}

    for part in ["middle","ring","highfloor"]:
        if tier and tier in SENTIMENT_POOLS:
            pool = SENTIMENT_POOLS[tier][part]
            result[part] = {"true": pick(rng, pool), "false": pick(rng, pool)}
        else:
            d = SENTIMENT_DEFAULT[part]
            result[part] = {"true": pick(rng, d["true"]), "false": pick(rng, d["false"])}

    cp = CLUSTER_POOLS.get(primary_cluster, CLUSTER_DEFAULT)
    result["ground_floor"] = {"true": pick(rng, cp["ground_true"]), "false": pick(rng, cp["ground_false"])}
    result["rooftop"]      = {"true": pick(rng, cp["roof_true"]),   "false": pick(rng, cp["roof_false"])}

    for key in result:
        vals = result[key]
        corner_pool = list({vals["true"], vals["false"]})
        result[key]["corner"] = pick(rng, corner_pool)

    return result

def apply_params(raw):
    area   = raw.get("area_ping") or 0
    price  = raw.get("price_wan") or 0
    layout = raw.get("layout") or ""
    width_depth     = 2 if area<25 else (4 if area<=50 else 6)
    height          = 4 if price<1000 else (6 if price<=2000 else 9)
    upper_floors    = 1 if re.search(r'套房|[1-2]\s*房',layout) else (3 if re.search(r'[3-9]\s*房|以上',layout) else 0)
    rings_frequency = 2 if raw.get("has_elevator") is True else 4
    gap_frequency   = 1 if raw.get("has_parking")  is True else 2
    return dict(width=width_depth, depth=width_depth, height=height,
                upper_floors=upper_floors,
                rings_frequency_h=rings_frequency, rings_frequency_v=rings_frequency,
                gap_frequency=gap_frequency)

PROMPT = """你是台灣售屋傳單解析引擎。仔細分析圖片，只回傳 JSON，不要任何其他文字或 markdown。

擷取以下所有欄位（找不到的填 null）：
{
  "title": "建案標題或物件名稱",
  "address": "地址",
  "phone": "電話",
  "area_ping": 坪數數字（取權狀坪數）,
  "land_ping": 土地坪數數字,
  "floor_count": 樓層數數字,
  "price_wan": 售價萬元數字,
  "layout": "格局描述如3房2廳",
  "has_elevator": true/false/null,
  "has_parking": true/false/null,
  "property_type": "透天/大樓/公寓/套房/店面/華廈",
  "agent": "房仲業者名稱",
  "surrounding_desc": "周邊環境描述（原文）",
  "ad_slogan": "美化標語（原文）",
  "material_desc": "建材描述或null",
  "condition_desc": "屋況描述或null",
  "color": "傳單背景顏色（中文）",
  "notes": "備註或特殊說明",
  "sentiment_score": 0到1數字（廣告文字正向承諾強度：0-0.5低 0.5-0.8中 0.8以上高）,
  "sentiment_reason": "一句話說明情感分數原因",
  "primary_cluster": "交通導向或機能導向或投資導向或景觀導向（只選最主要一個）",
  "cluster_confidence": 0到100數字,
  "cluster_reason": "一句話說明為何是這個導向"
}
判斷語意導向：交通導向=捷運/公車，機能導向=生活機能/學區，投資導向=增值/稀有/傳家，景觀導向=視野/公園/河景"""

async def call_claude(b64:str, mt:str):
    if not ANTHROPIC_API_KEY: raise HTTPException(500,"ANTHROPIC_API_KEY 未設定")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(ANTHROPIC_URL,
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-sonnet-4-5","max_tokens":1500,"messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":mt,"data":b64}},
                {"type":"text","text":PROMPT}
            ]}]})
    if r.status_code!=200: raise HTTPException(500,f"Claude API 錯誤 {r.status_code}: {r.text[:300]}")
    text = re.sub(r"```json|```","",r.json()["content"][0]["text"]).strip()
    try: return json.loads(text)
    except: raise HTTPException(500,f"JSON解析失敗: {text[:200]}")

def build_response(raw, seed: int):
    score   = raw.get("sentiment_score")
    cluster = raw.get("primary_cluster","")
    tier    = resolve_tier(score)
    return {
        "seed": seed,
        "raw": raw,
        "record": {k: raw.get(k) for k in ["title","address","phone","area_ping","land_ping",
            "floor_count","price_wan","layout","has_elevator","has_parking","property_type",
            "agent","surrounding_desc","ad_slogan","material_desc","condition_desc","color","notes"]},
        "params":           apply_params(raw),
        "facade_selection": build_facade(score, cluster, seed),
        "meta": {
            "sentiment_score":     score,
            "sentiment_tier":      tier,
            "sentiment_tier_label":{"low":"低正向","mid":"中正向","high":"高正向"}.get(tier,"未知"),
            "sentiment_reason":    raw.get("sentiment_reason",""),
            "primary_cluster":     cluster,
            "cluster_confidence":  raw.get("cluster_confidence",0),
            "cluster_reason":      raw.get("cluster_reason",""),
        },
        # 給前端用：顯示這個傳單所有可能的組合數
        "possible_combinations": _count_combinations(score, cluster),
        "machine_scores": calc_machine_scores(
            build_facade(score, cluster, seed),
            apply_params(raw)
        ),
        "interpretation": generate_interpretation(
            {k: raw.get(k) for k in ["title","address","phone","area_ping","land_ping",
                "floor_count","price_wan","layout","has_elevator","has_parking","property_type",
                "agent","surrounding_desc","ad_slogan","material_desc","condition_desc","color",
                "notes","location_desc"]},
            apply_params(raw),
            build_facade(score, cluster, seed),
            {"sentiment_tier": tier,
             "primary_cluster": cluster,
             "cluster_confidence": raw.get("cluster_confidence",0),
             "cluster_reason": raw.get("cluster_reason",""),}
        ),
    }

def _count_combinations(score, cluster):
    tier = resolve_tier(score)
    count = 1
    for part in ["middle","ring","highfloor"]:
        if tier and tier in SENTIMENT_POOLS:
            n = len(SENTIMENT_POOLS[tier][part])
            count *= n * n  # true × false
        else:
            count *= 4
    cp = CLUSTER_POOLS.get(cluster, CLUSTER_DEFAULT)
    count *= len(cp["ground_true"]) * len(cp["ground_false"])
    count *= len(cp["roof_true"])   * len(cp["roof_false"])
    return count

@app.get("/")
async def index(): return HTMLResponse(open("index.html").read())

@app.get("/assembler")
async def assembler(): return HTMLResponse(open("assembler.html").read())

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/parse")
async def parse_flyer(file: UploadFile = File(...), seed: Optional[int] = Form(None)):
    if not file.content_type.startswith("image/"): raise HTTPException(400,"請上傳圖片")
    data = await file.read()
    s = seed if seed is not None else random.randint(0, 999999)
    raw = await call_claude(base64.standard_b64encode(data).decode(), file.content_type)
    return build_response(raw, s)

@app.post("/parse/base64")
async def parse_base64(payload: dict):
    img = payload.get("image")
    if not img: raise HTTPException(400,"缺少 image 欄位")
    seed = payload.get("seed")
    s = seed if seed is not None else random.randint(0, 999999)
    raw = await call_claude(img, payload.get("media_type","image/jpeg"))
    return build_response(raw, s)

@app.post("/reseed")
async def reseed(payload: dict):
    """
    不重新分析傳單，只換 seed 重新隨機立面組合。
    Body: { "raw": {...}, "seed": 42 }   (seed 省略則隨機產生)
    給前端「刷新組合」按鈕用，不消耗 API 用量。
    """
    raw  = payload.get("raw")
    if not raw: raise HTTPException(400,"缺少 raw 欄位")
    seed = payload.get("seed")
    s = seed if seed is not None else random.randint(0, 999999)
    return build_response(raw, s)

@app.get("/parts-manifest")
def parts_manifest():
    """
    回傳所有立面零件的清單與機器評分，供 Unity/Three.js 一次性下載清單。
    每個零件的 url 指向 /parts/part_XX.glb
    """
    parts = []
    for i in range(16):
        info = PART_MACHINE.get(i, {})
        desc = PART_DESC.get(i, {})
        parts.append({
            "id": i,
            "filename": f"part_{i:02d}.glb",
            "url": f"/parts/part_{i:02d}.glb",
            "label": desc.get("label", f"#{i}"),
            "feel": desc.get("feel", ""),
            "bai": info.get("bai"),
            "end": info.get("end"),
            "ssi": info.get("ssi"),
            "aei": info.get("aei"),
        })
    return {"parts": parts, "part_role": PART_ROLE}

