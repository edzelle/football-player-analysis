using FootballDataReader.Data.Entities;
using FootballDataReader.Data.Queries;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data
{
    public class FootballContext : DbContext
    {
        public FootballContext(DbContextOptions<FootballContext> options)
            : base(options)
        {
        }

        public DbSet<Player> Players { get; set; }

        public DbSet<PlayerPlaysForTeam> PlayerPlaysForTeam { get; set; }

        public DbSet<PlayerReceivingStats> PlayerReceivingStats { get; set; }

        public DbSet<PlayerSeason> PlayerSeason { get; set; }

        public DbSet<Team> Teams { get; set; }

        public DbSet<TeamNameQuery> TeamNameQuery { get; set; }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            base.OnModelCreating(builder);
            builder.Entity<Player>().HasKey(x => new { x.Id });
            builder.Entity<PlayerPlaysForTeam>().HasKey(x => new { x.PlayerId, x.TeamId, x.EffectiveDate });
            builder.Entity<PlayerPlaysForTeam>().HasOne(ppt => ppt.Player).WithMany(player => player.PlayerPlaysForTeams);
            builder.Entity<PlayerPlaysForTeam>().HasOne(ppt => ppt.Team).WithMany(team => team.PlayerPlaysForTeams);
            builder.Entity<PlayerReceivingStats>().HasKey(x => new { x.Year, x.PlayerId });
            builder.Entity<PlayerReceivingStats>().HasOne(playerReceivingStats => playerReceivingStats.Player).WithMany(player => player.PlayerReceivingStats);
            builder.Entity<PlayerSeason>().HasKey(x => new { x.Year, x.PlayerId });
            builder.Entity<PlayerSeason>().HasOne(season => season.Player).WithMany(player => player.Seasons);
            builder.Entity<Team>().HasKey(x => new { x.Id });
            builder.Entity<TeamNameQuery>().HasNoKey();
        }
    }
}
