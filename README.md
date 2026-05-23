# 售屋傳單解析 API

售屋傳單圖片 → Claude Vision 解析 → 建築生成參數 + 立面模型選擇

## 本機測試

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-xxxxxxx
uvicorn main:app --reload
```

打開 http://localhost:8000/docs 可以看到互動式 API 文件並直接測試上傳。

## API 端點

### POST /parse
上傳圖片檔案（multipart/form-data）

```bash
curl -X POST http://localhost:8000/parse \
  -F "file=@傳單.jpg"
```

### POST /parse/base64
傳入 base64 字串，供 Quest 3 Unity app 使用

```bash
curl -X POST http://localhost:8000/parse/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "media_type": "image/jpeg"}'
```

## 回傳格式範例

```json
{
  "raw": {
    "area_ping": 110.23,
    "floor_count": 4,
    "price_wan": 7880,
    "layout": "透天店面",
    "has_elevator": null,
    "has_parking": null,
    "property_type": "透天",
    "location_desc": "龍山寺捷運",
    "ad_text": "稀有透天傳家寶 觀光夜市人潮多",
    "sentiment_score": 0.85,
    "semantic_cluster": ["交通導向", "投資導向"]
  },
  "params": {
    "width_depth": 6,
    "height": 9,
    "upper_floors": 0,
    "rings_frequency": 4,
    "gap_frequency": 2
  },
  "facade_selection": {
    "middle":      { "true": 8,  "false": 9  },
    "ring":        { "true": 10, "false": 0  },
    "highfloor":   { "true": 8,  "false": 13 },
    "ground_floor":{ "true": 7,  "false": 15 },
    "rooftop":     { "true": 12, "false": 14 }
  },
  "meta": {
    "sentiment_score": 0.85,
    "sentiment_tier": "sentiment_high",
    "semantic_cluster": ["交通導向", "投資導向"]
  }
}
```

## 部署到 Railway

1. 把這個資料夾推上 GitHub（新建一個 repo）
2. 去 railway.app，用 GitHub 登入
3. New Project → Deploy from GitHub repo → 選這個 repo
4. 在 Variables 裡加入 `ANTHROPIC_API_KEY=sk-ant-xxxxxxx`
5. 部署完成後得到一個 `https://xxxx.railway.app` 網址
6. Unity / Quest 3 打這個網址即可

## Unity C# 呼叫範例

```csharp
IEnumerator ParseFlyer(Texture2D screenshot)
{
    byte[] imageBytes = screenshot.EncodeToJPG();
    string base64 = System.Convert.ToBase64String(imageBytes);

    string json = JsonUtility.ToJson(new ParseRequest {
        image = base64,
        media_type = "image/jpeg"
    });

    using var req = new UnityWebRequest(API_URL + "/parse/base64", "POST");
    req.uploadHandler   = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
    req.downloadHandler = new DownloadHandlerBuffer();
    req.SetRequestHeader("Content-Type", "application/json");

    yield return req.SendWebRequest();

    if (req.result == UnityWebRequest.Result.Success)
    {
        BuildingData data = JsonUtility.FromJson<BuildingData>(req.downloadHandler.text);
        BuildingGenerator.Instance.Generate(data);
    }
}
```
