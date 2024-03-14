using FootballDataReader.Data.Config;
using FootballDataReader.Logic.IService;
using FootballDataReader.Logic.Service;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NLog;
using NLog.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader
{
    class Program
    {
        static void Main(string[] args)
        {
            IServiceCollection services = new ServiceCollection();
            Startup startup = new Startup();
            startup.ConfigureServices(services);
            var config = new ConfigurationBuilder().SetBasePath(Path.GetDirectoryName(Directory.GetParent(Environment.CurrentDirectory).FullName)).AddJsonFile("appsettings.json").Build();


            var logger = LogManager.Setup()
                                   .SetupExtensions(ext => ext.RegisterConfigSettings(config))
                                   .GetCurrentClassLogger();


            services
                .AddLogging(loggingBuilder =>
                {
                    // configure Logging with NLog
                    loggingBuilder.ClearProviders();
                    loggingBuilder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Debug);
                    loggingBuilder.AddNLog(config);
                })
                .AddSingleton<IFootballService, FootballService>()
                .AddSingleton<HttpClient>()
                .BuildServiceProvider();

            IServiceProvider serviceProvider = services.BuildServiceProvider();

          
            logger.Debug("Starting Application");

            var directory = services.RegisterConfiguration<DataDirectorySettings>(config.GetSection("LocalData"));

            var footballService = serviceProvider.GetService<IFootballService>();


            RunAsync(footballService, directory).Wait();


        }

        private static async Task RunAsync(IFootballService footballService, DataDirectorySettings directory)
        {
            //await footballService.ProcessNFLReceiverDataAndSaveToFile(path);
            //await footballService.AddPlayerClusters(path);
            //await footballService.ProcessRookieReceiverDataAndSaveToFile(directory.WRCollegeDirectory);
            //await footballService.LoadCollegeApiDataToDataBase();

            // TODO: Smooth EFF-LIM dates for player_plays_for_team

            await footballService.AddPlayerClusters(directory.ClusterDirectory);

            await footballService.CalculateYearTurnedPro();
            await footballService.AddPlayerAges();

        }
    }

}
