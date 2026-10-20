// Time the character-level ONNX voice (jq/sherpa-vits-tts-lug-eng) with ONNX Runtime on Android.
//   tts_bench MODEL_DIR [cpu|nnapi|xnnpack] [repeat] [model_file]
// Recipe (as in the Sunbird tutor app): lowercase, one id per character, no blanks, scales 0.667 1.0 0.8.
#include <chrono>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string mode = argc > 2 ? argv[2] : "cpu";
  int repeat = argc > 3 ? atoi(argv[3]) : 3;
  std::string file = argc > 4 ? argv[4] : "vits-lug-eng.fp32.onnx";
  std::map<std::string, int64_t> tok;  // symbol -> id (single bytes; the model uses ASCII letters and space)
  {
    std::ifstream f(dir + "/tokens.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      size_t p = line.rfind(' ');
      std::string sym = line.substr(0, p);
      if (sym.empty()) sym = " ";
      tok[sym] = std::stoll(line.substr(p + 1));
    }
  }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tts");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(4);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (mode == "nnapi") {
    OrtStatus* st = OrtSessionOptionsAppendExecutionProvider_Nnapi(so, 0);
    if (st) { std::cerr << "NNAPI unavailable: " << Ort::GetApi().GetErrorMessage(st) << "\n"; return 2; }
  } else if (mode == "xnnpack") {
    so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "4"}});
  }
  auto t0 = clk::now();
  Ort::Session session(env, (dir + "/" + file).c_str(), so);
  printf("%s [%s] load: %.2fs\n", file.c_str(), mode.c_str(), since(t0));

  const char* in_names[] = {"input", "input_lengths", "scales"};
  const char* out_names[] = {"output"};
  struct S { const char* lang; const char* text; };
  std::vector<S> sentences = {
      {"eng", "hello, how are you today?"},
      {"eng", "the weather is very nice this morning and we are going to the market"},
      {"lug", "oli otya, nsanyuse okukulaba leero"},
      {"lug", "ndi mukozi wa gavumenti era nnyumba yange eri e kampala"}};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (auto& s : sentences) {
    std::vector<int64_t> ids;
    for (char c : std::string(s.text)) {
      std::string k(1, (char)std::tolower((unsigned char)c));
      auto it = tok.find(k);
      if (it != tok.end()) ids.push_back(it->second);
    }
    int64_t len = (int64_t)ids.size();
    float scales[3] = {0.667f, 1.0f, 0.8f};
    int64_t shape[2] = {1, len}, lshape[1] = {1}, sshape[1] = {3};
    double audio_s = 0;
    std::vector<double> times;
    for (int r = 0; r < repeat + 1; r++) {  // first run is an untimed warm-up
      Ort::Value inputs[3] = {
          Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape, 2),
          Ort::Value::CreateTensor<int64_t>(mem, &len, 1, lshape, 1),
          Ort::Value::CreateTensor<float>(mem, scales, 3, sshape, 1)};
      auto t = clk::now();
      auto out = session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 1);
      double dt = since(t);
      size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
      audio_s = n / 22050.0;
      if (r > 0) times.push_back(dt);
    }
    std::ostringstream o;
    for (size_t i = 0; i < times.size(); i++) o << (i ? ", " : "") << times[i] << "s";
    printf("%s %zu chars, %.1fs audio: %s\n", s.lang, ids.size(), audio_s, o.str().c_str());
  }
  return 0;
}
