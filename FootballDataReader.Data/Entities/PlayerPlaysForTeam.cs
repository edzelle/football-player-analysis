using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Entities
{
    [Table("player_plays_for_team", Schema = "football")]
    public class PlayerPlaysForTeam
    {
        [Column("player_id")]
        public int PlayerId { get; set; }

        [Column("team_id")]
        public int TeamId { get; set; }

        [Column("player_name")]
        public string PlayerName { get; set; }

        [Column("effective_date")]
        public DateTime EffectiveDate { get; set; }

        [Column("limit_date")]
        public DateTime? LimitDate { get; set; }

        [ForeignKey("PlayerId")]
        public virtual Player Player { get; set; }

        [ForeignKey("TeamId")]
        public virtual Team Team { get; set; }
    }
}
