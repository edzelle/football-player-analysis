using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Models.CFB
{
    public class PPAUsageDto
    {
        public decimal? All { get; set; }

        public decimal? Pass { get; set; }

        public decimal? Rush { get; set; }

        public decimal? FirstDown { get; set; }

        public decimal? SecondDown { get; set; }

        public decimal? ThirdDown { get; set; }

        public decimal? StandardDowns { get; set; }

        public decimal? PassingDowns { get; set; }
    }
}
