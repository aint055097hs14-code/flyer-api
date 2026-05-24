import base64, json, random, re, os
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Optional

app = FastAPI(title="售屋傳單解析 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

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
