using Microsoft.AspNetCore.Mvc;
using NeuralNetwork.Grpc;
using WebAnimalAnalyzer.Models;

namespace WebAnimalAnalyzer.Controllers
{
    public class HomeController : Controller
    {
        private readonly AnimalRecognizer.AnimalRecognizerClient _grpcClient;
        private readonly ILogger<HomeController> _logger;

        public HomeController(AnimalRecognizer.AnimalRecognizerClient grpcClient, ILogger<HomeController> logger)
        {
            _grpcClient = grpcClient;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View(new AnalysisViewModel());
        }

        [HttpPost]
        public async Task<IActionResult> Index(AnalysisViewModel model)
        {
            if (model.UploadedImage == null || model.UploadedImage.Length == 0)
            {
                ModelState.AddModelError("UploadedImage", "Пожалуйста, выберите картинку.");
                return View(model);
            }

            try
            {
                byte[] imageBytes;
                using (var memoryStream = new MemoryStream())
                {
                    await model.UploadedImage.CopyToAsync(memoryStream);
                    imageBytes = memoryStream.ToArray();

                    string base64User = Convert.ToBase64String(imageBytes);
                    model.UserImageBase64 = $"data:{model.UploadedImage.ContentType};base64,{base64User}";
                }

                var request = new ImageRequest
                {
                    ImageData = Google.Protobuf.ByteString.CopyFrom(imageBytes)
                };

                var response = await _grpcClient.IdentifySpeciesAsync(request);

                model.IsAnalyzed = true;
                model.SpeciesName = response.SpeciesName;
                model.Confidence = response.Confidence;
                model.SimilarImagePath = response.SimilarImagePath;

                if (response.SimilarImageData != null && !response.SimilarImageData.IsEmpty)
                {
                    string base64Similar = Convert.ToBase64String(response.SimilarImageData.ToByteArray());
                    model.SimilarImageBase64 = $"data:image/jpeg;base64,{base64Similar}";
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Ошибка при обращении к нейросети");
                model.ErrorMessage = "Не удалось связаться с нейросетью. Убедитесь, что Python-сервер запущен.";
            }

            return View(model);
        }
    }
}
