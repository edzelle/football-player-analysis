using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Entities
{
    [Table("player_season", Schema = "football")]
    public class PlayerSeason
    {
        [Column("year")]
        public DateTime Year { get; set; }

        [Column("player_id")]
        public int PlayerId { get; set; }

        [Column("player_age")]
        public int? PlayerAge { get; set; }

        [Column("games_played")]
        public int? GamesPlayed { get; set; }

        [Column("games_started")]
        public int? GamesStarted { get; set; }

        [Column("is_college_season")]
        public bool? IsCollegeSeason { get; set; }

        [Column("cfdb_season_added")]
        public bool? CFDBSeasonAdded { get; set; }


        [ForeignKey("PlayerId")]
        public virtual Player Player { get; set; }
    }
}
