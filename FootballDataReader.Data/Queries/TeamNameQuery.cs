using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Queries
{
    public class TeamNameQuery
    {
        [Column("TeamName")]
        public string TeamName { get; set; }
    }
}
