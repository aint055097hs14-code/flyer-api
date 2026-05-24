import base64, json, random, re, os
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="售屋傳單解析 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# ── 立面模型池（依照對應參數規則.docx 完整規則）──────────────
# 情感分數 → 中間/環狀/高樓層 的 true & false 編號池（同池隨機各取一）
SENTIMENT_POOLS = {
    "low":  {"middle":[1,2,3,12,14], "ring":[1,2,3,12,14], "highfloor":[1,2,3,12,14]},
    "mid":  {"middle":[0,10,12],     "ring":[0,1,10,12,14],"highfloor":[0,10,12]},
    "high": {"middle":[0,8,9],       "ring":[0,8,9,10],    "highfloor":[0,8,13]},
}
# default（無分數時）
SENTIMENT_DEFAULT = {
    "middle":    {"true":0,      "false":[0,12]},
    "ring":      {"true":1,      "false":[0,2,12]},
    "highfloor": {"true":0,      "false":[0,12]},
}

# 語意群集 → 一樓/屋頂 的 true & false 編號池
CLUSTER_POOLS = {
    "交通導向": {"ground_true":[7,13,15],  "ground_false":[0,5,6,7,13,15], "roof_true":[12,14], "roof_false":[12,14]},
    "機能導向": {"ground_true":[9,11],     "ground_false":[0,5,6,7,9,11],  "roof_true":[12,14], "roof_false":[12,14]},
    "投資導向": {"ground_true":[13],       "ground_false":[0,13],           "roof_true":[10],    "roof_false":[10]},
    "景觀導向": {"ground_true":[13,9],     "ground_false":[0,9],            "roof_true":[10,13,14],"roof_false":[10,13,14]},
}
CLUSTER_DEFAULT = {"ground_true":[7],"ground_false":[0],"roof_true":[14],"roof_false":[14]}

def pick(pool):
    if isinstance(pool, list): return random.choice(pool) if pool else 0
    return pool

def resolve_tier(s):
    if s is None: return None
    return "low" if s<=0.5 else ("mid" if s<=0.8 else "high")

def build_facade(sentiment_score, primary_cluster):
    tier = resolve_tier(sentiment_score)
    result = {}

    # 中間、環狀、高樓層 — true/false 從同一個池各取一個
    for part in ["middle","ring","highfloor"]:
        if tier and tier in SENTIMENT_POOLS:
            pool = SENTIMENT_POOLS[tier][part]
            result[part] = {"true": pick(pool), "false": pick(pool)}
        else:
            d = SENTIMENT_DEFAULT[part]
            result[part] = {"true": pick(d["true"]), "false": pick(d["false"])}

    # 一樓、屋頂 — 由語意群集決定
    cp = CLUSTER_POOLS.get(primary_cluster, CLUSTER_DEFAULT)
    result["ground_floor"] = {"true": pick(cp["ground_true"]), "false": pick(cp["ground_false"])}
    result["rooftop"]      = {"true": pick(cp["roof_true"]),   "false": pick(cp["roof_false"])}

    # 轉角 — 從該部位 true+false 現有編號合集隨機取
    for part_key, facade_key in [("ground_floor","ground_floor"),("middle","middle"),
                                   ("ring","ring"),("highfloor","highfloor"),("rooftop","rooftop")]:
        vals = result[facade_key]
        corner_pool = list({vals["true"], vals["false"]})
        result[facade_key]["corner"] = pick(corner_pool)

    return result

def apply_params(raw):
    area    = raw.get("area_ping") or 0
    price   = raw.get("price_wan") or 0
    layout  = raw.get("layout") or ""
    floors  = raw.get("floor_count") or 0

    width_depth     = 2 if area<25 else (4 if area<=50 else 6)
    height          = 4 if price<1000 else (6 if price<=2000 else 9)
    upper_floors    = 1 if re.search(r'套房|[1-2]\s*房',layout) else (3 if re.search(r'[3-9]\s*房|以上',layout) else 0)
    rings_frequency = 2 if raw.get("has_elevator") is True else 4
    gap_frequency   = 1 if raw.get("has_parking")  is True else 2

    return dict(
        width=width_depth, depth=width_depth, height=height,
        upper_floors=upper_floors,
        rings_frequency_h=rings_frequency,
        rings_frequency_v=rings_frequency,
        gap_frequency=gap_frequency,
    )

PROMPT = """你是台灣售屋傳單解析引擎。仔細分析圖片，只回傳 JSON，不要任何其他文字或 markdown。

擷取以下所有欄位（找不到的填 null）：

{
  "title": "建案標題或物件名稱",
  "address": "地址",
  "phone": "電話",
  "area_ping": 坪數數字（取權狀坪數，找不到null）,
  "land_ping": 土地坪數數字（找不到null）,
  "floor_count": 樓層數數字,
  "price_wan": 售價萬元數字（租金填null）,
  "layout": "格局描述如3房2廳",
  "has_elevator": true/false/null,
  "has_parking": true/false/null,
  "property_type": "透天/大樓/公寓/套房/店面/華廈",
  "agent": "房仲業者名稱",
  "surrounding_desc": "周邊環境描述（原文）",
  "ad_slogan": "美化標語（原文）",
  "material_desc": "建材描述（原文或null）",
  "condition_desc": "屋況描述（原文或null）",
  "color": "傳單背景顏色（中文）",
  "notes": "備註或特殊說明",
  "sentiment_score": 0到1數字（廣告文字正向承諾強度：0-0.5低 0.5-0.8中 0.8以上高）,
  "sentiment_reason": "一句話說明情感分數原因",
  "primary_cluster": "交通導向或機能導向或投資導向或景觀導向（只選一個最主要的）",
  "cluster_confidence": 0到100數字,
  "cluster_reason": "一句話說明為何是這個導向"
}

判斷語意導向標準：
- 交通導向：強調捷運/公車/交通便利
- 機能導向：強調生活機能/市場/學區/醫院
- 投資導向：強調增值/出租/投報率/稀有/傳家
- 景觀導向：強調視野/公園/河景/山景"""

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

def build_response(raw):
    score   = raw.get("sentiment_score")
    cluster = raw.get("primary_cluster","")
    tier    = resolve_tier(score)
    return {
        "record": {
            "title":            raw.get("title"),
            "address":          raw.get("address"),
            "phone":            raw.get("phone"),
            "area_ping":        raw.get("area_ping"),
            "land_ping":        raw.get("land_ping"),
            "floor_count":      raw.get("floor_count"),
            "price_wan":        raw.get("price_wan"),
            "layout":           raw.get("layout"),
            "has_elevator":     raw.get("has_elevator"),
            "has_parking":      raw.get("has_parking"),
            "property_type":    raw.get("property_type"),
            "agent":            raw.get("agent"),
            "surrounding_desc": raw.get("surrounding_desc"),
            "ad_slogan":        raw.get("ad_slogan"),
            "material_desc":    raw.get("material_desc"),
            "condition_desc":   raw.get("condition_desc"),
            "color":            raw.get("color"),
            "notes":            raw.get("notes"),
        },
        "params":           apply_params(raw),
        "facade_selection": build_facade(score, cluster),
        "meta": {
            "sentiment_score":    score,
            "sentiment_tier":     tier,
            "sentiment_tier_label": {"low":"低正向","mid":"中正向","high":"高正向"}.get(tier,"未知"),
            "sentiment_reason":   raw.get("sentiment_reason",""),
            "primary_cluster":    cluster,
            "cluster_confidence": raw.get("cluster_confidence",0),
            "cluster_reason":     raw.get("cluster_reason",""),
        }
    }

@app.get("/")
async def index():
    return HTMLResponse(open("index.html").read())

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/parse")
async def parse_flyer(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): raise HTTPException(400,"請上傳圖片")
    data = await file.read()
    return build_response(await call_claude(base64.standard_b64encode(data).decode(), file.content_type))

@app.post("/parse/base64")
async def parse_base64(payload: dict):
    img = payload.get("image")
    if not img: raise HTTPException(400,"缺少 image 欄位")
    return build_response(await call_claude(img, payload.get("media_type","image/jpeg")))
