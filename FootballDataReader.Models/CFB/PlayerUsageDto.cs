using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Models.CFB
{
    public class PlayerUsageDto
    {
        public string Season { get; set; }

        public string Id { get; set; }

        public UsageDto Usage { get; set; }
    }

}
