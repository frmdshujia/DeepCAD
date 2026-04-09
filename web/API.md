# 后端 API 接口说明

## 预测接口

**URL:** `POST /api/predict`

**请求方式:** `multipart/form-data`

**请求参数:**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| image | File | 是 | 眼底图像文件（JPG、PNG 等） |

**响应格式:** `application/json`

**成功响应示例:**
```json
{
  "probability": 0.52
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| probability | number | 冠心病风险概率，范围 0～1，数值越大风险越高 |

**失败响应:** 返回适当 HTTP 状态码（如 400、500）及错误信息。

---

## 前端配置

在 `app.js` 中修改 `API_BASE` 为后端地址，例如：

```javascript
const API_BASE = 'http://localhost:8000';  // 本地开发
// 或
const API_BASE = 'https://your-domain.com';  // 生产环境
```

---

## 本地预览

需要用 HTTP 服务器访问，不能直接打开 `file://`（fetch 会报错）。

```bash
# Python
python3 -m http.server 8080

# Node.js (需安装 npx)
npx serve .

# 然后访问 http://localhost:8080
```
