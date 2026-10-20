// Tiny local voice server for the Sunflower phone web app.
//   tts_server MODEL_DIR [port]      POST /tts  {"text": "..."}  ->  audio/wav (22050 Hz, 16-bit mono)
// Same recipe as the Sunbird tutor app: lowercase, one id per character, no blanks, scales 0.667 1.0 0.8;
// text is split at punctuation and into chunks of 10 words, as in char_voice.py.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"

static std::map<std::string, int64_t> tok;
static Ort::Env* env;
static Ort::Session* sess;

static std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

static std::string json_text(const std::string& body) {  // value of "text": "..."
  size_t k = body.find("\"text\"");
  if (k == std::string::npos) return "";
  size_t q = body.find('"', body.find(':', k) + 1);
  std::string out;
  for (size_t i = q + 1; i < body.size(); i++) {
    char c = body[i];
    if (c == '"') break;
    if (c == '\\' && i + 1 < body.size()) {
      char e = body[++i];
      out += e == 'n' ? ' ' : e == 't' ? ' ' : e;
    } else out += c;
  }
  return out;
}

static std::vector<float> synth(const std::string& text) {
  // lower-case ASCII, then split at punctuation into chunks of at most 10 words
  std::string low;
  for (unsigned char c : text) low += (c < 0x80) ? (char)tolower(c) : (char)c;
  std::vector<std::string> parts(1);
  for (char c : low) {
    if (strchr(".,!?;:", c) && c) parts.emplace_back(); else parts.back() += c;
  }
  std::vector<float> audio;
  const char* in_names[] = {"input", "input_lengths", "scales"};
  const char* out_names[] = {"output"};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (auto& part : parts) {
    std::istringstream ws(part);
    std::vector<std::string> words;
    for (std::string w; ws >> w;) words.push_back(w);
    for (size_t i = 0; i < words.size(); i += 10) {
      std::string chunk;
      for (size_t j = i; j < words.size() && j < i + 10; j++) chunk += (j > i ? " " : "") + words[j];
      std::vector<int64_t> ids;
      for (auto& ch : utf8_chars(chunk)) { auto it = tok.find(ch); if (it != tok.end()) ids.push_back(it->second); }
      if (ids.empty()) continue;
      int64_t len = (int64_t)ids.size(), shape[2] = {1, len}, lshape[1] = {1}, sshape[1] = {3};
      float scales[3] = {0.667f, 1.0f, 0.8f};
      Ort::Value inputs[3] = {Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape, 2),
                              Ort::Value::CreateTensor<int64_t>(mem, &len, 1, lshape, 1),
                              Ort::Value::CreateTensor<float>(mem, scales, 3, sshape, 1)};
      auto out = sess->Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 1);
      const float* d = out[0].GetTensorData<float>();
      size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
      audio.insert(audio.end(), d, d + n);
      audio.insert(audio.end(), (size_t)(22050 * 0.15), 0.0f);
    }
  }
  return audio;
}

static void send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) { ssize_t n = send(fd, s.data() + off, s.size() - off, 0); if (n <= 0) break; off += n; }
}

static const char* CORS = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                          "Access-Control-Allow-Headers: Content-Type\r\nAccess-Control-Allow-Private-Network: true\r\n";

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  int port = argc > 2 ? atoi(argv[2]) : 8082;
  {
    std::ifstream f(dir + "/tokens.txt"); std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      size_t p = line.rfind(' ');
      std::string sym = line.substr(0, p); if (sym.empty()) sym = " ";
      tok[sym] = std::stoll(line.substr(p + 1));
    }
  }
  env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ttsd");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(4);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  sess = new Ort::Session(*env, (dir + "/vits-lug-eng.fp32.onnx").c_str(), so);
  int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(srv, (sockaddr*)&a, sizeof a) < 0 || listen(srv, 8) < 0) { perror("bind/listen"); return 1; }
  fprintf(stderr, "voice server ready on 127.0.0.1:%d\n", port);
  for (;;) {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    std::string req; char buf[8192]; size_t hdr_end = std::string::npos, want = 0;
    while (true) {
      ssize_t n = recv(fd, buf, sizeof buf, 0); if (n <= 0) break;
      req.append(buf, n);
      if (hdr_end == std::string::npos && (hdr_end = req.find("\r\n\r\n")) != std::string::npos) {
        size_t cl = req.find("Content-Length:"); if (cl == std::string::npos) cl = req.find("content-length:");
        want = hdr_end + 4 + (cl != std::string::npos ? (size_t)atol(req.c_str() + cl + 15) : 0);
      }
      if (hdr_end != std::string::npos && req.size() >= want) break;
    }
    std::string method = req.substr(0, req.find(' '));
    if (method == "OPTIONS") {
      send_all(fd, std::string("HTTP/1.1 204 No Content\r\n") + CORS + "Content-Length: 0\r\nConnection: close\r\n\r\n");
    } else if (req.find("GET /health") == 0) {
      send_all(fd, std::string("HTTP/1.1 200 OK\r\n") + CORS + "Content-Length: 2\r\nConnection: close\r\n\r\nok");
    } else if (req.find("POST /tts") == 0) {
      auto t0 = std::chrono::steady_clock::now();
      std::string text = json_text(req.substr(hdr_end + 4));
      std::vector<float> audio;
      try { audio = synth(text); } catch (const std::exception& e) { fprintf(stderr, "tts error: %s\n", e.what()); }
      std::string pcm(audio.size() * 2, 0);
      for (size_t i = 0; i < audio.size(); i++) {
        float v = audio[i] < -1 ? -1 : audio[i] > 1 ? 1 : audio[i];
        int16_t s = (int16_t)(v * 32767); memcpy(&pcm[i * 2], &s, 2);
      }
      uint32_t rate = 22050, byte_rate = rate * 2, data_len = pcm.size(), riff = 36 + data_len, fmt = 16;
      uint16_t pcm_fmt = 1, ch = 1, align = 2, bits = 16;
      std::string wav = "RIFF"; wav.append((char*)&riff, 4); wav += "WAVEfmt "; wav.append((char*)&fmt, 4);
      wav.append((char*)&pcm_fmt, 2); wav.append((char*)&ch, 2); wav.append((char*)&rate, 4); wav.append((char*)&byte_rate, 4);
      wav.append((char*)&align, 2); wav.append((char*)&bits, 2); wav += "data"; wav.append((char*)&data_len, 4); wav += pcm;
      double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      fprintf(stderr, "tts %.2fs for %zu chars, %.1fs audio\n", dt, text.size(), audio.size() / 22050.0);
      send_all(fd, std::string("HTTP/1.1 200 OK\r\n") + CORS + "Content-Type: audio/wav\r\nX-Synth-Seconds: " +
                       std::to_string(dt) + "\r\nContent-Length: " + std::to_string(wav.size()) + "\r\nConnection: close\r\n\r\n");
      send_all(fd, wav);
    } else {
      send_all(fd, std::string("HTTP/1.1 404 Not Found\r\n") + CORS + "Content-Length: 0\r\nConnection: close\r\n\r\n");
    }
    close(fd);
  }
}
