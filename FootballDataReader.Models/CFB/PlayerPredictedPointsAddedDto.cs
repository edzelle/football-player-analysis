using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Models.CFB
{
    public class PlayerPredictedPointsAddedDto
    {
        public int Season { get; set; }

        public string Id { get; set; }

        public string Name { get; set; }

        public int? CountablePlays { get; set; }

        public PPAUsageDto AveragePPA { get; set; }

        public PPAUsageDto TotalPPA { get; set; }
    }
}
