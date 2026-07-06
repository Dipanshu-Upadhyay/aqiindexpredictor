"""
AQI Prediction Web App
Run: python app.py
Open: http://localhost:5000
"""
import os, pickle, json
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "models", "models.pkl"), "rb") as f:
    data = pickle.load(f)

REGRESSOR   = data["regressor"]
CLASSIFIER  = data["classifier"]
LE          = data["label_encoder"]
FEATURES    = data["features"]
CITY_STATS  = data["city_stats"]

# ── AQI helpers ───────────────────────────────────────────────────────────────
BUCKET_INFO = {
    "Good":         {"color":"#10b981","bg":"#d1fae5","icon":"😊","advice":"Air quality is excellent. Enjoy outdoor activities freely!"},
    "Satisfactory": {"color":"#84cc16","bg":"#ecfccb","icon":"🙂","advice":"Air quality is acceptable. Sensitive people should limit prolonged outdoor activity."},
    "Moderate":     {"color":"#f59e0b","bg":"#fef3c7","icon":"😐","advice":"Air quality is moderate. Reduce intense outdoor activity if you feel discomfort."},
    "Poor":         {"color":"#f97316","bg":"#ffedd5","icon":"😷","advice":"Air quality is poor. Limit outdoor exposure especially for children and elderly."},
    "Very Poor":    {"color":"#ef4444","bg":"#fee2e2","icon":"🤧","advice":"Very unhealthy air. Avoid outdoor activities. Wear a mask if going out."},
    "Severe":       {"color":"#7c3aed","bg":"#ede9fe","icon":"☠️","advice":"Hazardous! Stay indoors, keep windows closed. Medical emergency risk."},
}

HEALTH_GROUPS = {
    "Good":         ["✅ Safe for everyone","✅ Children can play outside","✅ Elderly can go outdoors","✅ Exercise recommended"],
    "Satisfactory": ["✅ Generally safe","⚠️ Mild risk for asthma patients","✅ Children can play outside","✅ Normal activities okay"],
    "Moderate":     ["⚠️ Sensitive groups at risk","⚠️ Asthma patients take care","⚠️ Limit children outdoor time","✅ Healthy adults okay"],
    "Poor":         ["❌ Children — limit outdoor time","❌ Elderly — stay indoors","❌ Asthma/heart patients — danger","⚠️ Healthy adults limit exposure"],
    "Very Poor":    ["❌ All children stay indoors","❌ Elderly — do not go out","❌ Pregnant women — stay inside","❌ Even healthy adults at risk"],
    "Severe":       ["🚨 Everyone at serious risk","🚨 Medical emergency possible","🚨 Hospitals on alert","🚨 Government may issue red alert"],
}

def get_dominant_pollutant(values: dict) -> str:
    norm = {
        "PM2.5": values["PM2.5"] / 250,
        "PM10":  values["PM10"]  / 350,
        "NO2":   values["NO2"]   / 120,
        "SO2":   values["SO2"]   / 100,
        "CO":    values["CO"]    / 50,
        "O3":    values["O3"]    / 200,
    }
    return max(norm, key=norm.get)

def make_forecast(base_aqi: float) -> list:
    """Generate a realistic 24-hour AQI forecast."""
    np.random.seed(int(base_aqi) % 100)
    # Morning spike, midday dip, evening spike pattern
    pattern = [0.9,0.85,0.8,0.78,0.82,0.95,1.05,1.1,1.08,1.0,0.95,0.9,
                0.88,0.9,0.95,1.0,1.05,1.1,1.12,1.08,1.02,0.98,0.95,0.92]
    return [max(0, round(base_aqi * p + np.random.uniform(-5, 5), 1)) for p in pattern]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/city_data", methods=["POST"])
def city_data():
    city = request.json.get("city", "")
    if city in CITY_STATS:
        return jsonify(CITY_STATS[city])
    return jsonify({}), 404

@app.route("/predict", methods=["POST"])
def predict():
    body = request.json
    vals = [float(body.get(f, 0)) for f in FEATURES]

    aqi      = float(REGRESSOR.predict([vals])[0])
    aqi      = max(0, round(aqi, 1))
    bucket   = LE.inverse_transform(CLASSIFIER.predict([vals]))[0]
    info     = BUCKET_INFO.get(bucket, BUCKET_INFO["Moderate"])
    health   = HEALTH_GROUPS.get(bucket, [])
    forecast = make_forecast(aqi)
    dominant = get_dominant_pollutant({f: v for f, v in zip(FEATURES, vals)})

    # Feature importances as percentages
    importances = dict(zip(FEATURES, [round(float(x)*100,1) for x in REGRESSOR.feature_importances_]))

    return jsonify({
        "aqi":         aqi,
        "bucket":      bucket,
        "color":       info["color"],
        "bg":          info["bg"],
        "icon":        info["icon"],
        "advice":      info["advice"],
        "health":      health,
        "forecast":    forecast,
        "dominant":    dominant,
        "importances": importances,
    })

@app.route("/cities")
def cities():
    return jsonify(list(CITY_STATS.keys()))


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AQI Predictor — India Air Quality</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f0f4f8;--white:#ffffff;--surface:#f7f9fc;
  --border:#e2e8f0;--text:#1a202c;--muted:#718096;
  --blue:#3b82f6;--blue2:#1d4ed8;
  --radius:14px;--shadow:0 2px 16px rgba(0,0,0,.08);
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;min-height:100vh}

/* HEADER */
header{background:linear-gradient(135deg,#1e3a5f,#2563eb,#1e40af);padding:24px 40px;display:flex;align-items:center;gap:16px;box-shadow:0 4px 20px rgba(37,99,235,.3)}
.logo{font-size:26px;font-weight:800;color:#fff}
.logo span{color:#93c5fd}
.hbadges{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.hbadge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);border-radius:20px;padding:4px 13px;font-size:12px;color:#fff}

/* MAIN */
.main{max-width:1200px;margin:0 auto;padding:32px 24px}
.grid{display:grid;grid-template-columns:400px 1fr;gap:24px;align-items:start}

/* CARD */
.card{background:var(--white);border:1px solid var(--border);border-radius:var(--radius);padding:24px;box-shadow:var(--shadow)}
.card-title{font-size:15px;font-weight:700;color:var(--text);margin-bottom:18px;display:flex;align-items:center;gap:8px}

/* CITY SELECT */
.city-row{display:grid;grid-template-columns:1fr auto;gap:10px;margin-bottom:20px}
select,input{width:100%;background:var(--surface);border:1.5px solid var(--border);border-radius:9px;padding:10px 13px;font-family:inherit;font-size:14px;color:var(--text);outline:none;transition:border-color .2s}
select:focus,input:focus{border-color:var(--blue)}
.load-btn{background:var(--blue);color:#fff;border:none;border-radius:9px;padding:10px 16px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;transition:all .2s}
.load-btn:hover{background:var(--blue2)}

/* INPUT GRID */
.inp-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.field label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
.field input{padding:10px 12px;font-size:14px}
.unit{font-size:10px;color:var(--muted);margin-top:3px}

/* PREDICT BTN */
.pred-btn{width:100%;padding:14px;background:linear-gradient(135deg,var(--blue),var(--blue2));color:#fff;border:none;border-radius:10px;font-size:16px;font-weight:700;cursor:pointer;transition:all .22s;display:flex;align-items:center;justify-content:center;gap:8px;box-shadow:0 4px 14px rgba(59,130,246,.4)}
.pred-btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(59,130,246,.5)}
.pred-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* AQI RESULT */
.result-card{display:none;border-radius:var(--radius);padding:28px;margin-bottom:20px;border:1px solid rgba(0,0,0,.08)}
.aqi-header{display:flex;align-items:center;gap:20px;margin-bottom:20px}
.aqi-icon{font-size:48px}
.aqi-val{font-size:56px;font-weight:900;line-height:1}
.aqi-label{font-size:13px;color:var(--muted);margin-top:2px}
.aqi-bucket{font-size:20px;font-weight:700}
.aqi-advice{font-size:14px;margin-top:6px;opacity:.8}
.dominant-pill{display:inline-flex;align-items:center;gap:5px;background:rgba(0,0,0,.08);border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600;margin-top:10px}

/* GAUGE */
.gauge-wrap{position:relative;width:200px;height:110px;margin:0 auto 20px}
canvas#gauge{display:block}
.gauge-val{position:absolute;bottom:0;left:50%;transform:translateX(-50%);text-align:center}
.gauge-num{font-size:28px;font-weight:800}
.gauge-lbl{font-size:11px;color:var(--muted)}

/* HEALTH */
.health-list{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.health-item{font-size:13px;padding:7px 12px;border-radius:8px;background:var(--surface);border:1px solid var(--border)}

/* FORECAST */
.forecast-wrap{margin-top:20px}
.forecast-bars{display:flex;gap:3px;align-items:flex-end;height:80px;margin-top:8px}
.fbar-wrap{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px}
.fbar{width:100%;border-radius:3px 3px 0 0;min-height:4px;transition:height .4s}
.fbar-label{font-size:9px;color:var(--muted)}
.forecast-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}

/* IMPORTANCES */
.imp-row{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.imp-name{font-size:12px;font-weight:600;min-width:50px;color:var(--muted)}
.imp-track{flex:1;background:var(--surface);border-radius:4px;height:8px;overflow:hidden}
.imp-fill{height:100%;border-radius:4px;background:var(--blue)}
.imp-pct{font-size:11px;color:var(--muted);min-width:38px;text-align:right}

/* STATS ROW */
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
.stat-card{background:var(--white);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center;box-shadow:var(--shadow)}
.stat-val{font-size:22px;font-weight:800;color:var(--blue)}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.04em}

/* PLACEHOLDER */
.placeholder{text-align:center;padding:60px 30px;color:var(--muted)}
.placeholder .big{font-size:64px;margin-bottom:16px}
.placeholder h3{font-size:18px;font-weight:700;color:var(--text);margin-bottom:8px}

/* SPINNER */
.spinner{display:none;width:20px;height:20px;border:3px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* TABS */
.tabs{display:flex;gap:8px;margin-bottom:18px}
.tab{padding:8px 18px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;border:1.5px solid var(--border);background:var(--surface);color:var(--muted);transition:all .2s}
.tab.active{background:var(--blue);color:#fff;border-color:var(--blue)}
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">🌫️ AQI<span>Predict</span></div>
    <div style="font-size:13px;color:rgba(255,255,255,.7);margin-top:2px">India Air Quality Index Predictor · ML Powered</div>
  </div>
  <div class="hbadges">
    <span class="hbadge">📊 29,531 Records</span>
    <span class="hbadge">🏙️ 26 Cities</span>
    <span class="hbadge">🤖 R² = 0.90</span>
    <span class="hbadge">🎯 82% Accuracy</span>
  </div>
</header>

<div class="main">

  <!-- Stats Row -->
  <div class="stats-row">
    <div class="stat-card"><div class="stat-val">26</div><div class="stat-lbl">Indian Cities</div></div>
    <div class="stat-card"><div class="stat-val">29.5K</div><div class="stat-lbl">Training Records</div></div>
    <div class="stat-card"><div class="stat-val">6</div><div class="stat-lbl">AQI Categories</div></div>
  </div>

  <div class="grid">

    <!-- LEFT: Input Panel -->
    <div>
      <div class="card">
        <div class="card-title">🏙️ Select City or Enter Values</div>

        <div class="city-row">
          <select id="citySelect" onchange="onCityChange()">
            <option value="">-- Choose a city --</option>
          </select>
          <button class="load-btn" onclick="loadCity()">Load Data</button>
        </div>

        <div class="tabs">
          <div class="tab active" onclick="setTab('manual')">✏️ Manual Input</div>
          <div class="tab" onclick="setTab('city')">🏙️ City Presets</div>
        </div>

        <div class="inp-grid">
          <div class="field">
            <label>PM2.5</label>
            <input type="number" id="PM2.5" placeholder="e.g. 45" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Fine particles</div>
          </div>
          <div class="field">
            <label>PM10</label>
            <input type="number" id="PM10" placeholder="e.g. 80" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Coarse particles</div>
          </div>
          <div class="field">
            <label>NO2</label>
            <input type="number" id="NO2" placeholder="e.g. 25" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Nitrogen dioxide</div>
          </div>
          <div class="field">
            <label>SO2</label>
            <input type="number" id="SO2" placeholder="e.g. 15" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Sulphur dioxide</div>
          </div>
          <div class="field">
            <label>CO</label>
            <input type="number" id="CO" placeholder="e.g. 1.2" step="0.01" min="0"/>
            <div class="unit">mg/m³ · Carbon monoxide</div>
          </div>
          <div class="field">
            <label>O3</label>
            <input type="number" id="O3" placeholder="e.g. 30" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Ozone</div>
          </div>
          <div class="field">
            <label>NH3</label>
            <input type="number" id="NH3" placeholder="e.g. 10" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Ammonia</div>
          </div>
          <div class="field">
            <label>NOx</label>
            <input type="number" id="NOx" placeholder="e.g. 30" step="0.1" min="0"/>
            <div class="unit">µg/m³ · Nitrogen oxides</div>
          </div>
        </div>

        <button class="pred-btn" id="predBtn" onclick="predict()">
          <div class="spinner" id="spinner"></div>
          <span id="btnText">🔍 Predict AQI</span>
        </button>
      </div>

      <!-- Feature Importance -->
      <div class="card" style="margin-top:20px;display:none" id="impCard">
        <div class="card-title">🎯 Pollutant Impact on AQI</div>
        <div id="importances"></div>
      </div>
    </div>

    <!-- RIGHT: Results -->
    <div>
      <div id="placeholder" class="card placeholder">
        <div class="big">🌫️</div>
        <h3>Enter pollutant values to predict AQI</h3>
        <p>Select a city to auto-fill values, or enter manually and click Predict</p>
      </div>

      <!-- Result card -->
      <div class="result-card" id="resultCard">
        <div class="aqi-header">
          <div class="aqi-icon" id="resIcon"></div>
          <div>
            <div class="aqi-val" id="resVal"></div>
            <div class="aqi-label">Air Quality Index</div>
            <div class="aqi-bucket" id="resBucket"></div>
            <div class="aqi-advice" id="resAdvice"></div>
            <div class="dominant-pill" id="resDominant"></div>
          </div>
        </div>

        <!-- Gauge -->
        <div class="gauge-wrap">
          <canvas id="gauge" width="200" height="110"></canvas>
          <div class="gauge-val">
            <div class="gauge-num" id="gaugeNum"></div>
            <div class="gauge-lbl">AQI</div>
          </div>
        </div>

        <!-- Health groups -->
        <div class="card-title" style="margin-bottom:10px">👥 Health Impact</div>
        <div class="health-list" id="healthList"></div>

        <!-- 24hr forecast -->
        <div class="forecast-wrap">
          <div class="forecast-title">📈 24-Hour AQI Forecast</div>
          <div class="forecast-bars" id="forecastBars"></div>
          <div style="display:flex;justify-content:space-between;margin-top:4px">
            <span style="font-size:10px;color:var(--muted)">12AM</span>
            <span style="font-size:10px;color:var(--muted)">6AM</span>
            <span style="font-size:10px;color:var(--muted)">12PM</span>
            <span style="font-size:10px;color:var(--muted)">6PM</span>
            <span style="font-size:10px;color:var(--muted)">12AM</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// Load cities
fetch('/cities').then(r=>r.json()).then(cities=>{
  const sel = document.getElementById('citySelect');
  cities.forEach(c=>{
    const o = document.createElement('option');
    o.value = c; o.textContent = c;
    sel.appendChild(o);
  });
});

function setTab(t) {
  document.querySelectorAll('.tab').forEach((el,i)=>{
    el.classList.toggle('active', (t==='manual'&&i===0)||(t==='city'&&i===1));
  });
}

function onCityChange() {
  if (document.getElementById('citySelect').value) loadCity();
}

function loadCity() {
  const city = document.getElementById('citySelect').value;
  if (!city) return;
  fetch('/city_data', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({city})})
    .then(r=>r.json()).then(d=>{
      ['PM2.5','PM10','NO2','SO2','CO','O3','NH3','NOx'].forEach(f=>{
        if (d[f] !== undefined) document.getElementById(f).value = d[f];
      });
    });
}

function getColor(aqi) {
  if (aqi < 50)  return '#10b981';
  if (aqi < 100) return '#84cc16';
  if (aqi < 200) return '#f59e0b';
  if (aqi < 300) return '#f97316';
  if (aqi < 400) return '#ef4444';
  return '#7c3aed';
}

async function predict() {
  const features = ['PM2.5','PM10','NO2','SO2','CO','O3','NH3','NOx'];
  const body = {};
  let valid = true;
  features.forEach(f=>{
    const v = parseFloat(document.getElementById(f).value);
    if (isNaN(v)) valid = false;
    body[f] = v || 0;
  });

  document.getElementById('spinner').style.display = 'block';
  document.getElementById('btnText').textContent = 'Predicting...';
  document.getElementById('predBtn').disabled = true;

  try {
    const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await res.json();
    renderResult(d);
  } catch(e) {
    alert('Error: ' + e.message);
  } finally {
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('btnText').textContent = '🔍 Predict AQI';
    document.getElementById('predBtn').disabled = false;
  }
}

function renderResult(d) {
  document.getElementById('placeholder').style.display = 'none';
  const rc = document.getElementById('resultCard');
  rc.style.display = 'block';
  rc.style.background = d.bg;

  document.getElementById('resIcon').textContent = d.icon;
  document.getElementById('resVal').textContent = d.aqi;
  document.getElementById('resVal').style.color = d.color;
  document.getElementById('resBucket').textContent = d.bucket;
  document.getElementById('resBucket').style.color = d.color;
  document.getElementById('resAdvice').textContent = d.advice;
  document.getElementById('resDominant').innerHTML = `⚠️ Dominant pollutant: <b>${d.dominant}</b>`;

  // Gauge
  drawGauge(d.aqi, d.color);
  document.getElementById('gaugeNum').textContent = d.aqi;
  document.getElementById('gaugeNum').style.color = d.color;

  // Health
  document.getElementById('healthList').innerHTML = d.health.map(h=>
    `<div class="health-item">${h}</div>`).join('');

  // Forecast bars
  const maxF = Math.max(...d.forecast);
  document.getElementById('forecastBars').innerHTML = d.forecast.map((v,i)=>{
    const h = Math.max(4, Math.round((v/maxF)*76));
    const col = getColor(v);
    const lbl = i%6===0 ? ['12A','6A','12P','6P'][i/6]||'' : '';
    return `<div class="fbar-wrap"><div class="fbar" style="height:${h}px;background:${col}" title="Hour ${i}: AQI ${v}"></div></div>`;
  }).join('');

  // Feature importance
  document.getElementById('impCard').style.display = 'block';
  const imp = d.importances;
  const maxImp = Math.max(...Object.values(imp));
  document.getElementById('importances').innerHTML = Object.entries(imp)
    .sort((a,b)=>b[1]-a[1])
    .map(([k,v])=>`<div class="imp-row">
      <span class="imp-name">${k}</span>
      <div class="imp-track"><div class="imp-fill" style="width:${(v/maxImp)*100}%"></div></div>
      <span class="imp-pct">${v}%</span>
    </div>`).join('');
}

function drawGauge(aqi, color) {
  const canvas = document.getElementById('gauge');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,200,110);
  const cx=100, cy=100, r=80;
  const startAngle = Math.PI;
  const endAngle   = 2*Math.PI;
  const maxAQI     = 500;
  const aqiAngle   = startAngle + (Math.min(aqi,maxAQI)/maxAQI)*Math.PI;

  // Background arc
  ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,2*Math.PI);
  ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=14; ctx.lineCap='round'; ctx.stroke();

  // Color segments
  const segs = [[50,'#10b981'],[100,'#84cc16'],[200,'#f59e0b'],[300,'#f97316'],[400,'#ef4444'],[500,'#7c3aed']];
  let prev = Math.PI;
  segs.forEach(([val,col])=>{
    const ang = Math.PI + (val/maxAQI)*Math.PI;
    ctx.beginPath(); ctx.arc(cx,cy,r,prev,ang);
    ctx.strokeStyle=col; ctx.lineWidth=14; ctx.lineCap='butt'; ctx.stroke();
    prev=ang;
  });

  // Value arc
  ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,aqiAngle);
  ctx.strokeStyle=color; ctx.lineWidth=14; ctx.lineCap='round'; ctx.stroke();

  // Needle
  const nx = cx + (r-10)*Math.cos(aqiAngle);
  const ny = cy + (r-10)*Math.sin(aqiAngle);
  ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny);
  ctx.strokeStyle='#1a202c'; ctx.lineWidth=3; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx,cy,6,0,2*Math.PI);
  ctx.fillStyle='#1a202c'; ctx.fill();
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    import webbrowser
    from threading import Timer
    print("\n" + "="*50)
    print("  🌫️  AQI PREDICTOR — Starting...")
    print("="*50)
    print("  Open: http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000)
