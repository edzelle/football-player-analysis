using FootballDataReader.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using NLog.Extensions.Logging;
using System;
using System.IO;

namespace FootballDataReader
{
    public class Startup
    {
        public IConfiguration Configuration { get; }

        public Startup()
        {
            var builder = new ConfigurationBuilder().SetBasePath(Path.GetDirectoryName(Directory.GetParent(Environment.CurrentDirectory).FullName)).AddJsonFile("appsettings.json");
            Configuration = builder.Build();
        }

        public void ConfigureServices(IServiceCollection services)
        {
            var databaseSettings = new DatabaseSettings(); 
            if (!Configuration.GetSection("Database").Exists())
            {
                throw new Exception("Database does not exist in appsettings");
            }
            databaseSettings = services.RegisterConfiguration<DatabaseSettings>(Configuration.GetSection("Database"));
            string connectionString = String.Format("Server={0}; User Id={1}; Database={2}; Port={3}; Password={4};Include Error Detail=true;", databaseSettings.Host, databaseSettings.User, databaseSettings.DataBase, databaseSettings.Port, databaseSettings.Password);
            services.AddDbContext<FootballContext>(options =>
                options.UseNpgsql(connectionString));

            services.AddCollegeFootballApiClient(Configuration.GetSection("CFLClient"));

            services.AddLogging(loggingBuilder => {
                loggingBuilder.AddNLog("nlog.config");
            });
        }
    }
}
