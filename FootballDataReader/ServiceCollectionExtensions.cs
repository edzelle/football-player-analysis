using FootballDataReader.Client;
using FootballDataReader.Data.Config;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System;

namespace FootballDataReader
{
    public static class ServiceCollectionExtensions
    {
        public static TConfig RegisterConfiguration<TConfig>(this IServiceCollection services, IConfiguration configuration) where TConfig : class, new()
        {
            TConfig val = new TConfig();
            configuration.Bind(val);
            services.AddSingleton(val);
            return val;
        }

        public static void AddCollegeFootballApiClient(this IServiceCollection services, IConfigurationSection configurationSection)
        {
            var clientSettings = services.RegisterConfiguration<CollegeFootballClientSettings>(configurationSection);
            services.AddHttpClient("CollegeFootball_Client", httpClient =>
            {
                httpClient.BaseAddress = new Uri(clientSettings.Url);
                httpClient.DefaultRequestHeaders.Add("Authorization", string.Format("Bearer {0}", clientSettings.ApiKey));
            });

            services.AddSingleton<CollegeFootballHttpClient>();

        }
    }
}
