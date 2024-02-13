using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Entities
{
    [Table("teams", Schema = "football")]
    public class Team
    {
        [Column("id")]
        public int Id { get; set; }

        [Column("team_name")]
        public string TeamName { get; set; }

        public virtual ICollection<PlayerPlaysForTeam> PlayerPlaysForTeams { get; set; }
    }
}
