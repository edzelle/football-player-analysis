using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Entities
{
    [Table("players", Schema = "football")]
    public class Player
    {
        [Column("id")]
        public int Id { get; set; }

        [Column("name")]
        public string Name { get; set; }

        [Column("year_turned_pro")]
        public DateTime? YearTurnedPro { get; set; }

        [Column("position")]
        public string Position { get; set; }

        [Column("auxiliary_positions")]
        public string AuxiliaryPositions { get; set; }

        [Column("height")]
        public int? Height { get; set; }

        [Column("weight")]
        public int? Weight { get; set; }

        [Column("cfdb_player_id")]
        public int? CFDBPlayerId { get; set; }

        public virtual ICollection<PlayerSeason> Seasons { get; set; } 

        public virtual ICollection<PlayerReceivingStats> PlayerReceivingStats { get; set; }

        public virtual ICollection<PlayerPlaysForTeam> PlayerPlaysForTeams { get; set; }

    }
}
