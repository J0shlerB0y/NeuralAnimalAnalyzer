FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY ["WebAnimalAnalyzer.csproj", "./"]
RUN dotnet restore "WebAnimalAnalyzer.csproj"

COPY . .
RUN dotnet publish "WebAnimalAnalyzer.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 8080

ENTRYPOINT ["dotnet", "WebAnimalAnalyzer.dll"]