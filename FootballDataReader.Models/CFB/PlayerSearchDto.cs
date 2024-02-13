using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Models
{
    public class PlayerSearchDto
    {
        public int Id { get; set; }

        public string Name { get; set; }

        public string Team { get; set; }

        public int? Height { get; set; }

        public int? Weight { get; set; }

        public string Position { get; set; }
    }
}
