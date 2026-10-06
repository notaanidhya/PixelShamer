import axios from "axios";

const getBaseUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof window !== "undefined" && window.location.port === "5173") {
    return "http://localhost:8000";
  }
  return "";
};

const API_BASE_URL = getBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 min for large video uploads
});

// Separate client with longer timeout for video uploads
const videoApiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 min for video upload + inference
});

export const getSessionId = () => {
  if (typeof window === "undefined") return "anon_session";
  let sid = localStorage.getItem("smart_city_session_id");
  if (!sid) {
    sid = `sess_${Math.random().toString(36).substring(2, 10)}_${Date.now().toString(36)}`;
    localStorage.setItem("smart_city_session_id", sid);
  }
  return sid;
};

// Automatically attach session header for client isolation
apiClient.interceptors.request.use((config) => {
  config.headers["X-Session-ID"] = getSessionId();
  return config;
});

videoApiClient.interceptors.request.use((config) => {
  config.headers["X-Session-ID"] = getSessionId();
  return config;
});

export const formatLocalTimestamp = (isoString, type = "time") => {
  if (!isoString) return "—";
  let clean = String(isoString);
  if (!clean.endsWith("Z") && !clean.includes("+") && !clean.slice(10).includes("-")) {
    clean = `${clean}Z`;
  }
  const d = new Date(clean);
  if (isNaN(d.getTime())) return isoString;
  
  if (type === "time") {
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  }
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit"
  });
};

export const getAssetUrl = (path) => {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};

export const checkHealth = async () => {
  const t0 = performance.now();
  const response = await apiClient.get("/api/health");
  const latency = Math.round(performance.now() - t0);
  return { ...response.data, latency };
};

export const analyzeImage = async (file) => {
  const formData = new FormData();
  formData.append("image", file);

  const sid = getSessionId();
  const response = await apiClient.post(`/api/analyze?session_id=${encodeURIComponent(sid)}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      "X-Session-ID": sid,
    },
  });
  return response.data;
};

export const getResults = async (page = 1, limit = 10, qualityLabel = null, scope = "session") => {
  const params = {
    page,
    limit,
    scope,
    session_id: getSessionId(),
  };
  if (qualityLabel && qualityLabel !== "ALL") {
    params.quality_label = qualityLabel;
  }
  const response = await apiClient.get("/api/results", { params });
  return response.data;
};

export const getPresetAnalysis = async (presetId) => {
  const sid = getSessionId();
  const response = await apiClient.get(`/api/presets/${presetId}?session_id=${encodeURIComponent(sid)}`);
  return response.data;
};

export const getResultById = async (id) => {
  const response = await apiClient.get(`/api/results/${id}`);
  return response.data;
};

export const deleteResult = async (id) => {
  await apiClient.delete(`/api/results/${id}`);
  return true;
};

// --- Deepfake Detection API Client Methods ---

export const analyzeDeepfake = async (file) => {
  const formData = new FormData();
  formData.append("image", file);

  const sid = getSessionId();
  const response = await apiClient.post(`/api/deepfake/analyze?session_id=${encodeURIComponent(sid)}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      "X-Session-ID": sid,
    },
  });
  return response.data;
};

export const getDeepfakeResults = async (page = 1, limit = 10, verdict = null, scope = "session") => {
  const params = {
    page,
    limit,
    scope,
    session_id: getSessionId(),
  };
  if (verdict && verdict !== "ALL") {
    params.verdict = verdict;
  }
  const response = await apiClient.get("/api/deepfake/results", { params });
  return response.data;
};

export const getDeepfakeResultById = async (id) => {
  const response = await apiClient.get(`/api/deepfake/results/${id}`);
  return response.data;
};

export const deleteDeepfakeResult = async (id) => {
  await apiClient.delete(`/api/deepfake/results/${id}`);
  return true;
};

// --- Video Deepfake Detection API Client Methods ---

export const VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"];
export const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"];

export const getFileMediaType = (file) => {
  const ext = file.name.split(".").pop().toLowerCase();
  if (VIDEO_EXTENSIONS.includes(ext)) return "video";
  return "image";
};

export const analyzeVideoDeepfake = async (file, onUploadProgress = null) => {
  const formData = new FormData();
  formData.append("video", file);

  const sid = getSessionId();
  const response = await videoApiClient.post(
    `/api/deepfake/analyze-video?session_id=${encodeURIComponent(sid)}`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
        "X-Session-ID": sid,
      },
      onUploadProgress: onUploadProgress
        ? (evt) => {
            if (evt.total) {
              const pct = Math.round((evt.loaded / evt.total) * 100);
              onUploadProgress(pct);
            }
          }
        : undefined,
    }
  );
  return response.data;
};

export const getVideoDeepfakeResults = async (page = 1, limit = 10, verdict = null, scope = "session") => {
  const params = {
    page,
    limit,
    scope,
    session_id: getSessionId(),
  };
  if (verdict && verdict !== "ALL") {
    params.verdict = verdict;
  }
  const response = await apiClient.get("/api/deepfake/history/videos", { params });
  return response.data;
};

export const getVideoDeepfakeResultById = async (id) => {
  const response = await apiClient.get(`/api/deepfake/videos/${id}`);
  return response.data;
};

export const deleteVideoDeepfakeResult = async (id) => {
  await apiClient.delete(`/api/deepfake/videos/${id}`);
  return true;
};

export default apiClient;
