using FootballDataReader.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Logic.IService
{
    public interface IFootballService
    {
        Task ProcessRookieReceiverDataAndSaveToFile(string path);

        Task ProcessNFLReceiverDataAndSaveToFile(string path);

        Task<PlayerSearchDto> GetCollegePlayerMetaData(string playerName, int playerId);

        Task LoadCollegeApiDataToDataBase();

        Task CalculateYearTurnedPro();
        Task AddPlayerAges();
    }
}
