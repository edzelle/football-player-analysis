using FootballDataReader.Data.Config;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Client
{
    public class CollegeFootballHttpClient
    {
        private readonly HttpClient _client;

        public CollegeFootballHttpClient(IHttpClientFactory factory)
        {
            _client = factory.CreateClient("CollegeFootball_Client");
        }

        public async Task<HttpResponseMessage> GetAsync(string route)
        {
            try
            {
                return await _client.GetAsync(route).ConfigureAwait(false);
            } catch (Exception ex)
            {
                throw ex;
            }
        }
    }
}
