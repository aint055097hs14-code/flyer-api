import anthropic
import base64
import json
import random
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI(title="售屋傳單解析 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── 立面模型池規則 ──────────────────────────────────────────────
# 情感分數對應每個部位的 true/false 模型編號池
FACADE_RULES = {
    "sentiment_low": {        # 0 ~ 0.5
        "middle":      {"true": [1,2,3,12,14], "false": [1,2,3,12,14]},
        "ring":        {"true": [1,2,3,12,14], "false": [1,2,3,12,14]},
        "highfloor":   {"true": [1,2,3,12,14], "false": [1,2,3,12,14]},
    },
    "sentiment_mid": {        # 0.5 ~ 0.8
        "middle":      {"true": [0,10,12],     "false": [0,10,12]},
        "ring":        {"true": [0,1,10,12,14],"false": [0,1,10,12,14]},
        "highfloor":   {"true": [0,10,12],     "false": [0,10,12]},
    },
    "sentiment_high": {       # 0.8 ~ 1.0
        "middle":      {"true": [0,8,9],       "false": [0,8,9]},
        "ring":        {"true": [0,8,9,10],    "false": [0,8,9,10]},
        "highfloor":   {"true": [0,8,13],      "false": [0,8,13]},
    },
}

# 語意群集對應一樓與屋頂的模型編號池
SEMANTIC_RULES = {
    "交通導向": {
        "ground_floor": {"true": [7,13,15], "false": [0,5,6,7,13,15]},
        "rooftop":      {"true": [12,14],   "false": [12,14]},
    },
    "機能導向": {
        "ground_floor": {"true": [9,11],    "false": [0,5,6,7,9,11]},
        "rooftop":      {"true": [12,14],   "false": [12,14]},
    },
    "投資導向": {
        "ground_floor": {"true": [13],      "false": [0,13]},
        "rooftop":      {"true": [10],      "false": [10]},
    },
    "景觀導向": {
        "ground_floor": {"true": [13,9],    "false": [0,9]},
        "rooftop":      {"true": [10,13,14],"false": [10,13,14]},
    },
}

DEFAULT_GROUND = {"true": [7], "false": [0]}
DEFAULT_ROOFTOP = {"true": [14], "false": [14]}


def pick(pool: list) -> int:
    """從 pool 隨機抽一個編號"""
    return random.choice(pool) if pool else 0


def resolve_sentiment_tier(score: float) -> str:
    if score <= 0.5:
        return "sentiment_low"
    elif score <= 0.8:
        return "sentiment_mid"
    else:
        return "sentiment_high"


def build_facade_selection(sentiment_score: float, semantic_clusters: list) -> dict:
    """根據情感分數和語意群集，決定每個部位的模型編號"""
    tier = resolve_sentiment_tier(sentiment_score)
    tier_rules = FACADE_RULES[tier]

    result = {}

    # 中間、環狀、高樓層 → 由情感分數決定
    for part in ["middle", "ring", "highfloor"]:
        pools = tier_rules[part]
        result[part] = {
            "true":  pick(pools["true"]),
            "false": pick(pools["false"]),
        }

    # 一樓、屋頂 → 由語意群集決定（取第一個有對應規則的群集）
    ground_pools = DEFAULT_GROUND
    rooftop_pools = DEFAULT_ROOFTOP
    for cluster in semantic_clusters:
        if cluster in SEMANTIC_RULES:
            ground_pools  = SEMANTIC_RULES[cluster]["ground_floor"]
            rooftop_pools = SEMANTIC_RULES[cluster]["rooftop"]
            break

    result["ground_floor"] = {
        "true":  pick(ground_pools["true"]),
        "false": pick(ground_pools["false"]),
    }
    result["rooftop"] = {
        "true":  pick(rooftop_pools["true"]),
        "false": pick(rooftop_pools["false"]),
    }

    return result


def apply_building_params(raw: dict) -> dict:
    """套用傳單數值 → 建築幾何參數"""
    area   = raw.get("area_ping") or 0
    price  = raw.get("price_wan") or 0
    layout = raw.get("layout") or ""

    # width / depth
    if area < 25:
        width_depth = 2
    elif area <= 50:
        width_depth = 4
    else:
        width_depth = 6

    # height
    if price < 1000:
        height = 4
    elif price <= 2000:
        height = 6
    else:
        height = 9

    # upper_floors
    if re.search(r'套房|[1-2]\s*房', layout):
        upper_floors = 1
    elif re.search(r'[3-9]\s*房|以上', layout):
        upper_floors = 3
    else:
        upper_floors = 0

    # rings_frequency
    has_elevator = raw.get("has_elevator")
    rings_frequency = 2 if has_elevator is True else 4

    # gap_frequency
    has_parking = raw.get("has_parking")
    gap_frequency = 1 if has_parking is True else 2

    return {
        "width_depth":      width_depth,
        "height":           height,
        "upper_floors":     upper_floors,
        "rings_frequency":  rings_frequency,
        "gap_frequency":    gap_frequency,
    }


PARSE_PROMPT = """你是台灣售屋傳單解析引擎。分析圖片並回傳 JSON，不要有任何其他文字或 markdown。

提取欄位：
- area_ping: 坪數（數字，取權狀坪數）
- floor_count: 樓層數（數字）
- price_wan: 售價萬元（數字）
- layout: 格局描述（字串）
- has_elevator: 有無電梯（true/false/null）
- has_parking: 有無車位（true/false/null）
- property_type: 物件類型（透天/大樓/公寓/套房/店面）
- location_desc: 地點描述
- ad_text: 所有廣告行銷文字

NLP 分析：
- sentiment_score: 0-1，居住價值正向承諾強度
  - 0~0.5：描述模糊、品質承諾不足
  - 0.5~0.8：一般市場描述
  - 0.8~1.0：強烈承諾舒適與價值
- semantic_cluster: 陣列，從 [交通導向, 機能導向, 投資導向, 景觀導向] 選適合的

回傳格式：
{
  "area_ping": 數字,
  "floor_count": 數字,
  "price_wan": 數字,
  "layout": "字串",
  "has_elevator": true/false/null,
  "has_parking": true/false/null,
  "property_type": "字串",
  "location_desc": "字串",
  "ad_text": "字串",
  "sentiment_score": 數字,
  "semantic_cluster": ["字串"]
}"""


@app.get("/")
def root():
    return {"status": "ok", "message": "售屋傳單解析 API 運行中"}


@app.post("/parse")
async def parse_flyer(file: UploadFile = File(...)):
    """
    上傳傳單圖片，回傳建築生成參數與立面模型選擇。
    
    Unity / Quest 3 端呼叫範例：
    POST /parse
    Content-Type: multipart/form-data
    file: <image bytes>
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "請上傳圖片檔案（JPG/PNG）")

    image_bytes = await file.read()
    image_b64   = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type  = file.content_type

    # 呼叫 Claude Vision
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_b64,
                    },
                },
                {"type": "text", "text": PARSE_PROMPT},
            ],
        }],
    )

    raw_text = message.content[0].text.strip()
    raw_text = re.sub(r"```json|```", "", raw_text).strip()

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Claude 回傳格式錯誤：{raw_text[:200]}")

    # 套用規則
    params          = apply_building_params(raw)
    facade_selection = build_facade_selection(
        raw.get("sentiment_score", 0.5),
        raw.get("semantic_cluster", []),
    )

    return {
        "raw":              raw,
        "params":           params,
        "facade_selection": facade_selection,
        "meta": {
            "sentiment_score": raw.get("sentiment_score"),
            "sentiment_tier":  resolve_sentiment_tier(raw.get("sentiment_score", 0.5)),
            "semantic_cluster": raw.get("semantic_cluster", []),
        }
    }


@app.post("/parse/base64")
async def parse_flyer_base64(payload: dict):
    """
    接受 base64 字串，給 Quest 3 Passthrough 截圖用。
    Body: { "image": "<base64 string>", "media_type": "image/jpeg" }
    """
    image_b64  = payload.get("image")
    media_type = payload.get("media_type", "image/jpeg")

    if not image_b64:
        raise HTTPException(400, "缺少 image 欄位")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_b64,
                    },
                },
                {"type": "text", "text": PARSE_PROMPT},
            ],
        }],
    )

    raw_text = message.content[0].text.strip()
    raw_text = re.sub(r"```json|```", "", raw_text).strip()

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Claude 回傳格式錯誤：{raw_text[:200]}")

    params           = apply_building_params(raw)
    facade_selection = build_facade_selection(
        raw.get("sentiment_score", 0.5),
        raw.get("semantic_cluster", []),
    )

    return {
        "raw":              raw,
        "params":           params,
        "facade_selection": facade_selection,
        "meta": {
            "sentiment_score":  raw.get("sentiment_score"),
            "sentiment_tier":   resolve_sentiment_tier(raw.get("sentiment_score", 0.5)),
            "semantic_cluster": raw.get("semantic_cluster", []),
        }
    }
