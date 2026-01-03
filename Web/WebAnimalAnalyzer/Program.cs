using NeuralNetwork.Grpc;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();

builder.Services.AddGrpcClient<AnimalRecognizer.AnimalRecognizerClient>(options =>
{
    var aiServiceUrl = Environment.GetEnvironmentVariable("AI_SERVICE_URL") ?? "http://localhost:50051";
    options.Address = new Uri(aiServiceUrl);
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();