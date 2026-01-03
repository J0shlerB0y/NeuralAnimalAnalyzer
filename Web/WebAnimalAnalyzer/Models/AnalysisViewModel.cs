namespace WebAnimalAnalyzer.Models
{
    public class AnalysisViewModel
    {
        public IFormFile? UploadedImage { get; set; }

        public bool IsAnalyzed { get; set; } = false;
        public string? SpeciesName { get; set; }
        public float Confidence { get; set; }

        public string? UserImageBase64 { get; set; }

        public string? SimilarImageBase64 { get; set; }
        public string? SimilarImagePath { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
